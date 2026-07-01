# FSSAI Indian Health Star Rating AI System

> A production-grade Streamlit application that evaluates the nutritional quality of Indian packaged foods and traditional cooked dishes against the official **FSSAI Indian Nutrition Rating (INR)** framework — combining deterministic rule-based scoring with optional ML-assisted prediction in a hybrid architecture.

---

## 🏗️ Hybrid Architecture Overview

The system employs a **two-track scoring pipeline** designed to eliminate non-monotonic anomalies that occur when relying solely on machine learning classifiers for nutrition scoring:

### Track 1 — Deterministic FSSAI Rule Engine (Primary)
All manual entry, dataset search, and traditional dish lookup paths route exclusively through the **strict FSSAI INR point-subtraction formula**, fully bypassing ML models:

- **Category I (Solid Foods):** Baseline risk points are accumulated for Energy (> 80 kcal threshold, 30 kcal intervals), Total Sugars (> 4.5 g threshold), Saturated Fat (> 1.0 g threshold), and Sodium (> 90 mg threshold), each capped at 10 points. A critical **11-point gatekeeper rule** governs positive nutrient offsets: below 11 baseline points, both Protein and Dietary Fibre offset scores reduce the final tally; at or above 11 baseline points, only the Fibre offset is permitted per the official FSSAI legal constraint.

- **Category II (Liquid Beverages):** A completely separate, steeper penalty ladder is applied — Energy > 60 kcal immediately triggers a 10-point penalty and Total Sugars > 13.6 g triggers a 10-point penalty, with combined scores >= 20 mapped to the mandatory 0.5-star floor.

### Track 2 — Machine Learning Classifier (Fallback / Vision AI Path)
When the **Vision AI OCR** module uploads and scans a physical Nutrition Information Panel (NIP), the extracted and user-verified values are passed into pre-trained `scikit-learn` classification models (`packaged_model.pkl`, `cooked_model.pkl`) for an AI-assisted star prediction. Liquid beverages always bypass ML and use the deterministic Track 1 rules regardless of which input path is active.

---

## ✨ Core Feature Set

### 📦 Packaged Foods Classifier
- **Search Food:** Autocomplete product lookup from the `packaged_foods_india.csv` dataset (500+ commercial Indian products). Extracts Energy, Saturated Fat, Total Sugars, Sodium, Protein, and Dietary Fibre per 100g and renders them in a structured nutrient matrix table. Accepts a Product Form toggle (Solid / Liquid) to route scoring correctly.
- **Vision AI NIP Scanner:** Upload a photograph of any Nutrition Information Panel. `EasyOCR` extracts raw text, a regex parser normalises OCR noise and maps nutritional values, and the user can verify and correct each field before final scoring.
- **Manual NIP Entry:** Six precision `st.number_input` fields — with hard boundaries and floating-point step increments — allow direct data entry for any product not in the dataset.

### 🍲 Traditional Cooked Dishes Classifier
- Full-text autocomplete search across the `Indian_Food_Nutrition_Processed.csv` dataset (curated Indian recipes: Garam Chai, Biryani, Dal Makhani, etc.).
- Side-by-side Product Form selector for correct liquid vs. solid scoring routing.
- Nutrient breakdown rendered in a dark-themed horizontal `st.table` matrix.

### 🧮 FSSAI INR Mathematical Engine
- `calculate_fssai_solid_score()` — complete solid food pipeline with the 11-point gatekeeper.
- `get_liquid_energy_points()` / `get_liquid_sugar_points()` — compact formula-based liquid ladders.
- `get_liquid_stars()` — strict star mapping with `score >= 20 → 0.5 Stars` mandatory floor.
- HFSS Safety Warning banner triggers when `Sodium / Energy > 1.0`.

### 💡 Smart Health Substitution Engine
When a scanned or searched product scores **1.5 Stars or lower**, the engine automatically scans the full dataset in real-time, applies the FSSAI formula to every candidate row, and surfaces up to **5 alternative products scoring 3.5 Stars or higher** in a visually distinctive dark glassmorphism recommendation panel.

### 📥 Dynamic PDF Report Exporter
Every workspace (Search, Upload, Manual, Cooked) includes a one-click `📥 Export Nutritional Report PDF` button powered by `fpdf2`. The generated PDF includes:
- Product name and form factor header
- HFSS safety alert banner (if triggered)
- Full 6-nutrient horizontal matrix table
- FSSAI Health Star Rating banner with ASCII star representation
- Final score, tier label (Excellent / Average / Poor / Very Poor), and official guideline footer

---

## 📁 Repository Structure

```
HealthStarRatingSystem/
│
├── app.py                  # Main Streamlit application entry point
├── train_model.py          # ML model training script (Phase 1)
├── requirements.txt        # Python dependency manifest
├── .gitignore              # Repository exclusion rules
├── README.md               # This file
│
├── models/                 # Pre-trained ML model binaries
│   ├── packaged_model.pkl
│   ├── cooked_model.pkl
│   └── fssai_model.pkl
│
└── Dataset/                # Source nutrition datasets
    ├── packaged_foods_india.csv          # 500+ commercial Indian packaged food items
    ├── Indian_Food_Nutrition_Processed.csv  # Traditional Indian recipe nutrition data
    └── Indian_Food_Nutrition_Labeled.csv    # Labeled training dataset for ML models
```

---

## 🚀 Local Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/HealthStarRatingSystem.git
cd HealthStarRatingSystem
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `easyocr` will automatically download its neural network weights (~100 MB) on first run. Ensure you have an active internet connection for the initial launch.

### Step 3 — (Optional) Retrain ML Models
If you wish to retrain the classifiers from the source datasets:
```bash
python train_model.py
```
This will regenerate `models/packaged_model.pkl` and `models/cooked_model.pkl`.

### Step 4 — Launch the Application
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501` in your default browser.

---

## 📊 FSSAI INR Star Mapping Reference

| Final Score | Health Stars |
|:-----------:|:------------:|
| ≤ 0         | ⭐⭐⭐⭐⭐ 5.0 |
| 1 – 2       | ⭐⭐⭐⭐½ 4.5 |
| 3 – 4       | ⭐⭐⭐⭐  4.0 |
| 5 – 6       | ⭐⭐⭐½  3.5 |
| 7 – 9       | ⭐⭐⭐   3.0 |
| 10 – 12     | ⭐⭐½   2.5 |
| 13 – 15     | ⭐⭐    2.0 |
| 16 – 18     | ⭐½    1.5 |
| 19          | ⭐     1.0 |
| ≥ 20        | ½     0.5 |

---

## 🧪 Verified Test Cases

| Product | Category | Energy | Sugars | Sat Fat | Sodium | Protein | Fibre | Expected Stars |
|---------|----------|--------|--------|---------|--------|---------|-------|---------------|
| Kwality Walls Cassata | Solid | 157 kcal | 13.3 g | 3.3 g | 32 mg | 3.5 g | 0 g | **4.0 ★** |
| Frooti Mango Juice | Liquid | 62.9 kcal | 15.6 g | — | — | — | — | **0.5 ★** |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend / App Framework | [Streamlit](https://streamlit.io) |
| Data Handling | Pandas, NumPy |
| ML Classification | scikit-learn (Random Forest / Gradient Boosting) |
| OCR Engine | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| PDF Generation | [fpdf2](https://py-pdf.github.io/fpdf2/) |
| Image Processing | Pillow |
| Model Serialisation | joblib |

---

## 📜 Regulatory Reference

This application implements the **FSSAI Food Safety and Standards (Labelling and Display) Regulations** — specifically the Indian Nutrition Rating (INR) system introduced under the FSS Act 2006. All scoring thresholds, category definitions, and star-to-score mappings are derived directly from the official FSSAI INR Technical Guidance Document.

---

## 📄 License

This project is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it with attribution.
