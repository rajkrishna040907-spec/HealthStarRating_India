import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: xgboost not installed. Falling back to Logistic Regression and Random Forest only.")

print("Phase 1: Data Engineering & Synthetic Labeling...")

DATA_DIR = r"D:\HealthStarRatingSystem\Dataset"
PACKAGED_PATH = os.path.join(DATA_DIR, "packaged_foods_india.csv")
COOKED_PATH = os.path.join(DATA_DIR, "Indian_Food_Nutrition_Processed.csv")

PACKAGED_MODEL_PATH = r"D:\HealthStarRatingSystem\packaged_model.pkl"
COOKED_MODEL_PATH = r"D:\HealthStarRatingSystem\cooked_model.pkl"

def load_data(path, is_packaged):
    df_raw = pd.read_csv(path)
    df = pd.DataFrame()
    
    if is_packaged:
        df['Energy'] = pd.to_numeric(df_raw['Calories_kcal'], errors='coerce').fillna(0.0)
        sat_fat = pd.to_numeric(df_raw['Saturated_Fat_g'], errors='coerce')
        tot_fat = pd.to_numeric(df_raw['Total_Fat_g'], errors='coerce')
        df['Sat_Fat'] = sat_fat.replace(0.0, np.nan).fillna(tot_fat).fillna(0.0)
        df['Sugar'] = pd.to_numeric(df_raw['Sugar_g'], errors='coerce').fillna(0.0)
        df['Sodium'] = pd.to_numeric(df_raw['Sodium_mg'], errors='coerce').fillna(0.0)
        df['Protein'] = pd.to_numeric(df_raw['Proteins_g'], errors='coerce').fillna(0.0)
        df['Fiber'] = pd.to_numeric(df_raw['Dietary_Fiber_g'], errors='coerce').fillna(0.0)
    else:
        df['Energy'] = pd.to_numeric(df_raw['Calories (kcal)'], errors='coerce').fillna(0.0)
        df['Sat_Fat'] = pd.to_numeric(df_raw['Fats (g)'], errors='coerce').fillna(0.0)
        df['Sugar'] = pd.to_numeric(df_raw['Free Sugar (g)'], errors='coerce').fillna(0.0)
        df['Sodium'] = pd.to_numeric(df_raw['Sodium (mg)'], errors='coerce').fillna(0.0)
        df['Protein'] = pd.to_numeric(df_raw['Protein (g)'], errors='coerce').fillna(0.0)
        df['Fiber'] = pd.to_numeric(df_raw['Fibre (g)'], errors='coerce').fillna(0.0)
        
    return df.dropna(subset=['Energy', 'Sat_Fat', 'Sugar', 'Sodium', 'Protein', 'Fiber'])

def calculate_fssai_stars(row):
    # Baseline Risk Points
    energy_kcal = row['Energy']
    sat_fat_g = row['Sat_Fat']
    sugar_g = row['Sugar']
    sodium_mg = row['Sodium']
    
    # Positive Nutrient Points
    protein_g = row['Protein']
    fiber_g = row['Fiber']
    
    # Energy
    if energy_kcal <= 80: energy_pts = 0
    elif energy_kcal <= 160: energy_pts = 1
    elif energy_kcal <= 240: energy_pts = 2
    elif energy_kcal <= 320: energy_pts = 3
    elif energy_kcal <= 400: energy_pts = 4
    else: energy_pts = 5
    
    # Saturated Fat
    if sat_fat_g <= 1.0: fat_pts = 0
    elif sat_fat_g <= 2.0: fat_pts = 1
    elif sat_fat_g <= 3.0: fat_pts = 2
    elif sat_fat_g <= 4.0: fat_pts = 3
    elif sat_fat_g <= 5.0: fat_pts = 4
    else: fat_pts = 5
    
    # Total Sugars
    if sugar_g <= 4.2: sugar_pts = 0
    elif sugar_g <= 8.4: sugar_pts = 1
    elif sugar_g <= 12.6: sugar_pts = 2
    elif sugar_g <= 16.8: sugar_pts = 3
    elif sugar_g <= 21.0: sugar_pts = 4
    else: sugar_pts = 5
    
    # Sodium
    if sodium_mg <= 90: sodium_pts = 0
    elif sodium_mg <= 180: sodium_pts = 1
    elif sodium_mg <= 270: sodium_pts = 2
    elif sodium_mg <= 360: sodium_pts = 3
    elif sodium_mg <= 450: sodium_pts = 4
    else: sodium_pts = 5
    
    baseline_pts = energy_pts + fat_pts + sugar_pts + sodium_pts
    
    # Fiber
    if fiber_g <= 3.0: fiber_pts = 0
    elif fiber_g <= 6.0: fiber_pts = 1
    elif fiber_g <= 9.0: fiber_pts = 2
    elif fiber_g <= 12.0: fiber_pts = 3
    elif fiber_g <= 15.0: fiber_pts = 4
    else: fiber_pts = 5
    
    # Protein
    if protein_g <= 1.5: protein_pts = 0
    elif protein_g <= 2.0: protein_pts = 1
    elif protein_g <= 2.5: protein_pts = 2
    elif protein_g <= 3.0: protein_pts = 3
    elif protein_g <= 5.0: protein_pts = 4
    else: protein_pts = 5
    
    positive_pts = fiber_pts + protein_pts
    
    net_score = baseline_pts - positive_pts
    
    # String mapping
    if net_score <= -1: return '5.0'
    elif net_score <= 2: return '4.5'
    elif net_score <= 5: return '4.0'
    elif net_score <= 8: return '3.5'
    elif net_score <= 12: return '3.0'
    elif net_score <= 16: return '2.5'
    elif net_score <= 20: return '2.0'
    elif net_score <= 24: return '1.5'
    elif net_score <= 28: return '1.0'
    else: return '0.5'

def run_tournament(df, dataset_name, output_path):
    print(f"\n--- Running Tournament for {dataset_name} ---")
    
    X = df[['Energy', 'Sat_Fat', 'Sugar', 'Sodium', 'Protein', 'Fiber']]
    y = df['FSSAI_Stars']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(eval_metric='mlogloss', random_state=42)
        
    best_acc = 0.0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc * 100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model
            
    print(f"Champion Model for {dataset_name}: {best_model_name} (Accuracy: {best_acc * 100:.2f}%)")
    
    # Save the model and label encoder together
    joblib.dump({'model': best_model, 'label_encoder': le}, output_path)
    print(f"Saved champion model to: {output_path}")

if __name__ == "__main__":
    df_packaged = load_data(PACKAGED_PATH, is_packaged=True)
    df_cooked = load_data(COOKED_PATH, is_packaged=False)
    
    df_packaged['FSSAI_Stars'] = df_packaged.apply(calculate_fssai_stars, axis=1)
    df_cooked['FSSAI_Stars'] = df_cooked.apply(calculate_fssai_stars, axis=1)
    
    run_tournament(df_packaged, "Packaged Foods", PACKAGED_MODEL_PATH)
    run_tournament(df_cooked, "Cooked Dishes", COOKED_MODEL_PATH)
