import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
import io
from PIL import Image

import easyocr

# ─────────────────────────────────────────────────────────────
# DATA & MODEL LOADERS
# ─────────────────────────────────────────────────────────────
# Determine base path relative to app.py location
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_reader()

@st.cache_data
def load_dataset():
    path = os.path.join(_BASE_DIR, "Dataset", "packaged_foods_india.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_cooked_dataset():
    path = os.path.join(_BASE_DIR, "Dataset", "Indian_Food_Nutrition_Processed.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_ml_models():
    models = {
        'Packaged Foods': None, 'Packaged Label Encoder': None,
        'Cooked Dishes': None,  'Cooked Label Encoder': None
    }
    pkg_path = os.path.join(_BASE_DIR, "models", "packaged_model.pkl")
    if os.path.exists(pkg_path):
        try:
            pkg_data = joblib.load(pkg_path)
            models['Packaged Foods'] = pkg_data['model']
            models['Packaged Label Encoder'] = pkg_data['label_encoder']
        except Exception as e:
            st.error(f"Error loading packaged model: {e}")

    ckd_path = os.path.join(_BASE_DIR, "models", "cooked_model.pkl")
    if os.path.exists(ckd_path):
        try:
            ckd_data = joblib.load(ckd_path)
            models['Cooked Dishes'] = ckd_data['model']
            models['Cooked Label Encoder'] = ckd_data['label_encoder']
        except Exception as e:
            st.error(f"Error loading cooked model: {e}")
    return models

models = load_ml_models()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Indian Health Star Rating AI",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 3rem;
        color: #2E86C1;
        text-align: center;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8F9F9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: bold;
        color: #273746;
    }
    .metric-label {
        font-size: 1rem;
        color: #5D6D7E;
    }
    .star-rating {
        font-size: 3rem;
        color: #F1C40F;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .stSelectbox label, .stSlider label {
        font-weight: 600;
        color: #34495E;
    }
    .alt-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border: 1px solid #00FFCC44;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        color: #e0e0e0;
    }
    .alt-card h4 {
        color: #00FFCC;
        margin: 0 0 6px 0;
        font-size: 1.05rem;
    }
    .alt-card span {
        font-size: 0.92rem;
        color: #b0b8c1;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# OCR HELPERS
# ─────────────────────────────────────────────────────────────

def validate_nip_content(text):
    """
    Loosened validator: accepts as valid if *any* of the broad keyword sets
    match -- handles heavy OCR noise where most keywords are garbled.
    """
    text_lower = text.lower()
    # Broad keyword clusters that cover common OCR corruptions
    clusters = [
        # energy / calories cluster
        ['energy', 'calori', 'kcal', 'kj', 'eheroy', 'eneroy', 'fhergy', 'kal'],
        # protein cluster
        ['protein', 'prot', 'protn'],
        # fat cluster
        ['fat', 'malfa', 'mlfat', 'saturated', 'sat fat', 'trans'],
        # carbohydrate / sugar cluster
        ['sugar', 'carbo', 'sugars', 'sucrose', 'sugara', 'sugarse'],
        # sodium / salt cluster
        ['sodium', 'salt', 'sowm', 'soiui', 'soiu', 'na'],
        # fibre cluster
        ['fibre', 'fiber', 'dietary'],
    ]
    found = sum(1 for cluster in clusters
                if any(kw in text_lower for kw in cluster))
    return found >= 2


def _normalise_ocr(text: str) -> str:
    """
    Applies character-level substitutions for the most common OCR errors
    seen on nutrition panel scans before regex matching.
    """
    # --- whole-word OCR mis-reads ---
    word_map = [
        # energy
        ('eheroy', 'energy'), ('eneroy', 'energy'), ('ehercy', 'energy'),
        ('fhergy', 'energy'), ('ehergy', 'energy'), ('enegry', 'energy'),
        ('enery', 'energy'),  ('enercy', 'energy'),
        # protein
        ('prot ', 'protein '), ('protn', 'protein'),
        # fat / saturated fat
        ('malfa', 'fat'),  ('mlfat', 'fat'), ('faт', 'fat'),
        ('saturateo', 'saturated'), ('satureated', 'saturated'),
        ('saturatcd', 'saturated'), ('saturatad', 'saturated'),
        # sugars
        ('sugarse', 'sugars'), ('sugara', 'sugars'), ('ugars', 'sugars'),
        ('sugarsa', 'sugars'), ('suqars', 'sugars'), ('su9ars', 'sugars'),
        ('sugrs', 'sugars'),
        # sodium
        ('soiui', 'sodium'), ('soiu', 'sodium'), ('sowm', 'sodium'),
        ('sodum', 'sodium'), ('sodiu', 'sodium'), ('sodi', 'sodium'),
        # fiber
        ('dietry', 'dietary'), ('fibr ', 'fiber '), ('fber', 'fiber'),
        # calories / kcal
        ('kcai', 'kcal'), ('kca1', 'kcal'), ('kcaI', 'kcal'), (' kal', ' kcal'),
        ('calori', 'calories'), ('cal ', 'calories '),
        # carbohydrate
        ('carbohydrat', 'carbohydrate'), ('carbohyd', 'carbohydrate'),
        # added
        ('aqded', 'added'), ('adoed', 'added'),
        # trans
        ('tran5', 'trans'), ('tran fat', 'trans fat'),
    ]
    t = text.lower()
    for wrong, right in word_map:
        t = t.replace(wrong, right)

    # --- character-level number OCR swaps ---
    char_map = str.maketrans({
        'O': '0', 'I': '1', 'l': '1', 'S': '5', 'Z': '2', 'G': '6',
    })
    # Only translate inside digit-adjacent contexts to avoid destroying words
    # We do a targeted digit-sequence repair after spacing:
    t = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', t)
    t = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', t)

    return t


def parse_nutritional_info(text: str) -> dict:
    """
    Robust NIP parser with four-layer extraction strategy:
      1. OCR normalisation (word + character swaps)
      2. Primary regex patterns (expanded alternations)
      3. Fuzzy fallback patterns (partial keyword stems)
      4. Decimal-shift correction for all nutrients
    """
    raw = text  # keep original for fallback scans
    text = _normalise_ocr(text)

    # ── Layer 1 & 2: Primary + expanded patterns ──────────────
    patterns = {
        'Energy': (
            r'(?:energy|calories|kcal|kj|eheroy|eneroy|enercy|enery)'
            r'[^\d]{0,25}?(\d+(?:[.,]\d+)?)'
        ),
        'Fat': (
            r'(?:total\s*fat|fat)'
            r'[^\\d]{0,15}(\\d+(?:[.,]\\d+)?)'
        ),
        'Sat_Fat': (
            r'(?:saturated|sat\.?\s*fat|sat\s+fat)'
            r'[^\d]{0,20}?(\d+(?:[.,]\d+)?)'
        ),
        'Sugar': (
            r'(?:total\s*sugars?|sugars?|sucrose)'
            r'[^\d]{0,20}?(\d+(?:[.,]\d+)?)'
        ),
        'Sodium': (
            r'(?:sodium|salt|sowm|soiu)'
            r'[^\d]{0,20}?(\d+(?:[.,]\d+)?)'
        ),
        'Protein': (
            r'(?:protein|protn)'
            r'[^\d]{0,20}?(\d+(?:[.,]\d+)?)'
        ),
        'Fiber': (
            r'(?:dietary\s*fi[bv]e?r|fi[bv]e?r|fibre)'
            r'[^\d]{0,20}?(\d+(?:[.,]\d+)?)'
        ),
    }

    results = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val_str = m.group(1).replace(',', '.')
            results[key] = float(val_str)
        else:
            results[key] = None   # mark as unresolved for fallback

    # ── Layer 3: Fuzzy stem fallbacks for unresolved keys ─────
    fuzzy_patterns = {
        'Energy':  r'(?:ener|cal)[^\d]{0,15}(\d{2,4}(?:[.,]\d+)?)',
        'Sat_Fat': r'(?:satur|sat)[^\d]{0,15}(\d+(?:[.,]\d+)?)',
        'Sugar':   r'(?:sug|ugar)[^\d]{0,15}(\d+(?:[.,]\d+)?)',
        'Sodium':  r'(?:sodi|sowm|sod)[^\d]{0,15}(\d+(?:[.,]\d+)?)',
        'Protein': r'(?:prot)[^\d]{0,15}(\d+(?:[.,]\d+)?)',
        'Fiber':   r'(?:fib|diet)[^\d]{0,15}(\d+(?:[.,]\d+)?)',
    }
    for key, pat in fuzzy_patterns.items():
        if results[key] is None:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                results[key] = float(m.group(1).replace(',', '.'))

    # ── Layer 4: Decimal-shift correction ─────────────────────
    # Nutrient ceiling reference (per 100g/ml)
    ceilings = {
        'Energy': 900.0,   # kcal
        'Sat_Fat': 100.0,
        'Sugar':   100.0,
        'Sodium': 3000.0,  # mg — do not scale sodium
        'Protein': 100.0,
        'Fiber':   100.0,
    }
    no_scale = {'Sodium', 'Energy'}  # these can legitimately be large

    for key in results:
        val = results[key]
        if val is None:
            results[key] = 0.0
            continue
        if key not in no_scale and val > ceilings[key]:
            while val > ceilings[key]:
                val /= 10.0
        results[key] = round(val, 2)

    return results


# ─────────────────────────────────────────────────────────────
# MODULE 1 — FSSAI RULE-BASED CALCULATION ENGINE
# ─────────────────────────────────────────────────────────────

def get_liquid_energy_points(energy):
    """Liquid beverage energy penalty — strict FSSAI ladder."""
    if energy <= 6:   return 0
    if energy > 60:   return 10
    return int(energy // 6)

def get_liquid_sugar_points(sugar):
    """Liquid beverage sugar penalty — strict FSSAI ladder."""
    if sugar <= 0.1:  return 0
    if sugar > 13.6:  return 10
    return int((sugar - 0.1) // 1.5) + 1

def get_liquid_stars(score):
    """Map liquid total score to FSSAI stars — score >= 20 floors to 0.5."""
    if score <= 0:   return 5.0
    elif score <= 2:  return 4.5
    elif score <= 4:  return 4.0
    elif score <= 6:  return 3.5
    elif score <= 9:  return 3.0
    elif score <= 12: return 2.5
    elif score <= 15: return 2.0
    elif score <= 18: return 1.5
    elif score < 20:  return 1.0
    else:             return 0.5   # score >= 20 → mandatory 0.5 floor

def calculate_fssai_solid_score(energy, sat_fat, sugar, sodium, protein, fiber):
    """
    Full FSSAI solid-food point calculation with the official 11-point
    gatekeeper rule for positive nutrient offsets.

    Returns (final_score, star_rating).
    """
    # --- Baseline risk points ---
    energy_pts  = min(10, max(0, int((energy - 80) // 30))) if energy > 80 else 0
    sugar_pts   = min(10, max(0, int((sugar  - 4.5) // 4.5))) if sugar  > 4.5 else 0
    sat_fat_pts = min(10, max(0, int((sat_fat - 1.0) // 1.0))) if sat_fat > 1.0 else 0
    sodium_pts  = min(10, max(0, int((sodium - 90)  // 90)))  if sodium > 90  else 0
    total_baseline = energy_pts + sugar_pts + sat_fat_pts + sodium_pts

    # --- Positive nutrient offsets ---
    protein_pts = min(5, max(0, int(protein // 1.6)))
    fiber_pts   = min(5, max(0, int(fiber   // 0.9)))

    # --- FSSAI 11-point gatekeeper ---
    # If baseline < 11, both protein & fiber offsets apply freely.
    # If baseline >= 11, protein points are forfeited — only fiber applies.
    if total_baseline < 11:
        total_positive = protein_pts + fiber_pts
    else:
        total_positive = fiber_pts   # protein offset locked out above threshold

    final_score = total_baseline - total_positive

    # --- Star mapping ---
    if final_score <= 0:   star = 5.0
    elif final_score <= 2: star = 4.5
    elif final_score <= 4: star = 4.0
    elif final_score <= 6: star = 3.5
    elif final_score <= 9: star = 3.0
    elif final_score <= 12: star = 2.5
    elif final_score <= 15: star = 2.0
    elif final_score <= 18: star = 1.5
    elif final_score < 20:  star = 1.0
    else:                   star = 0.5

    return final_score, star

def calculate_fssai_rating(energy, sat_fat, sugar, sodium, protein, fiber,
                            product_category="Solid Food"):
    """
    Top-level dispatcher: routes to the correct FSSAI calculation
    track based on product_category.
    Returns (final_score, star_rating).
    """
    if product_category == "Liquid Beverage":
        energy_pts = get_liquid_energy_points(energy)
        sugar_pts  = get_liquid_sugar_points(sugar)
        total      = energy_pts + sugar_pts
        return total, get_liquid_stars(total)
    else:
        return calculate_fssai_solid_score(energy, sat_fat, sugar, sodium, protein, fiber)


# ─────────────────────────────────────────────────────────────
# MODULE 2 — RATING DISPLAY
# ─────────────────────────────────────────────────────────────

def render_star_display(predicted_star, label="📐 Strict FSSAI Rule-Based Rating"):
    """Renders the star widget and health tier badge."""
    full_stars  = int(predicted_star)
    half_star   = 1 if predicted_star - full_stars >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    stars_html  = "★" * full_stars
    if half_star:
        stars_html += "⯨"
    stars_html += "☆" * empty_stars
    st.markdown(f'<h3 style="text-align:center;color:#34495E;">{label}</h3>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="star-rating">{stars_html} '
        f'<span style="font-size:1.5rem;color:#7F8C8D;">⭐ {predicted_star} / 5.0</span></div>',
        unsafe_allow_html=True
    )
    if predicted_star >= 4.0:
        st.success("🟢 Excellent Choice! This food item is highly nutritious.")
    elif predicted_star >= 2.5:
        st.warning("🟡 Average. Consume in moderation.")
    elif predicted_star >= 1.5:
        st.error("🔴 Poor Nutritional Value. Try to avoid or limit intake.")
    else:
        st.error("🚨 Very Poor. High health risk — strongly avoid.")

def display_fssai_rating(energy, sat_fat, sugar, sodium, protein, fiber,
                          product_category="Solid Food",
                          product_name="", show_hfss=True):
    """
    Full rating pipeline: score -> HFSS warning (gated) -> star display.
    Returns (final_score, star_rating) for downstream use.
    The HFSS banner is suppressed when predicted_star == 5.0 to prevent
    false alarms on naturally low-calorie healthy items (e.g. vegetable
    stock) where a tiny energy denominator artificially inflates the ratio.
    """
    # Score first — so the HFSS gate can check the actual star value
    final_score, predicted_star = calculate_fssai_rating(
        energy, sat_fat, sugar, sodium, protein, fiber, product_category
    )

    if show_hfss and energy > 0 and (sodium / energy) > 1.0 and predicted_star < 5.0:
        st.markdown(
            '<div style="background-color:#F8C471;padding:15px;border-radius:10px;'
            'color:#935116;margin-bottom:20px;text-align:center;font-weight:bold;">'
            '\u26a0\ufe0f HFSS Safety Warning: High Fat, Sugar & Salt detected! '
            '(Sodium-to-Energy ratio > 1.0)</div>',
            unsafe_allow_html=True
        )

    label = "\U0001f964 FSSAI Liquid Beverage Rating" \
        if product_category == "Liquid Beverage" \
        else "\U0001f4d0 Strict FSSAI Rule-Based Rating"

    render_star_display(predicted_star, label)
    return final_score, predicted_star


# ─────────────────────────────────────────────────────────────
# MODULE 2 — ML PREDICTION (used by Vision AI / Upload NIP path)
# ─────────────────────────────────────────────────────────────

def display_ml_prediction(energy, sat_fat, sugar, sodium, protein, fiber,
                           model_key, product_category):
    """
    ML-backed prediction for OCR/upload path.
    Liquid beverages always bypass ML and use FSSAI rules.
    """
    # Pre-compute the star so we can suppress HFSS banner on perfect scores
    _pre_score, _pre_star = calculate_fssai_rating(
        energy, sat_fat, sugar, sodium, protein, fiber, product_category
    )
    if energy > 0 and (sodium / energy) > 1.0 and _pre_star < 5.0:
        st.markdown(
            '<div style="background-color:#F8C471;padding:15px;border-radius:10px;'
            'color:#935116;margin-bottom:20px;text-align:center;font-weight:bold;">'
            '\u26a0\ufe0f HFSS Safety Warning: High Fat, Sugar & Salt detected! '
            '(Sodium-to-Energy ratio > 1.0)</div>',
            unsafe_allow_html=True
        )

    if product_category == "Liquid Beverage":
        energy_pts   = get_liquid_energy_points(energy)
        sugar_pts    = get_liquid_sugar_points(sugar)
        total        = energy_pts + sugar_pts
        predicted_star = get_liquid_stars(total)
        render_star_display(predicted_star, "🥤 FSSAI Liquid Beverage Rating")
    else:
        mdl = models.get(model_key)
        le  = models.get(f"{model_key.split(' ')[0]} Label Encoder")
        if mdl is not None and le is not None:
            features = pd.DataFrame(
                [[energy, sat_fat, sugar, sodium, protein, fiber]],
                columns=['Energy', 'Sat_Fat', 'Sugar', 'Sodium', 'Protein', 'Fiber']
            )
            pred_idx       = mdl.predict(features)[0]
            predicted_star = float(le.inverse_transform([pred_idx])[0])
            render_star_display(predicted_star, "🤖 AI Predicted Rating")
        else:
            st.error(f"ML model for '{model_key}' not found. Run the Phase 1 training script.")
            return None

    return predicted_star


# ─────────────────────────────────────────────────────────────
# MODULE 3 — SMART HEALTH SUBSTITUTION ENGINE
# ─────────────────────────────────────────────────────────────

PACKAGED_ENERGY_COL  = 'Calories_kcal'
PACKAGED_FAT_COL     = 'Total_Fat_g'
PACKAGED_SAT_FAT_COL = 'Saturated_Fat_g'
PACKAGED_SUGAR_COL   = 'Sugar_g'
PACKAGED_SODIUM_COL  = 'Sodium_mg'
PACKAGED_PROTEIN_COL = 'Proteins_g'
PACKAGED_FIBER_COL   = 'Dietary_Fiber_g'
PACKAGED_NAME_COL    = 'Item name'

COOKED_ENERGY_COL    = 'Calories (kcal)'
COOKED_FAT_COL       = 'Total_Fat (g)'
COOKED_SAT_FAT_COL   = 'Saturated_Fat (g)'
COOKED_SUGAR_COL     = 'Free Sugar (g)'
COOKED_SODIUM_COL    = 'Sodium (mg)'
COOKED_PROTEIN_COL   = 'Protein (g)'
COOKED_FIBER_COL     = 'Fibre (g)'
COOKED_NAME_COL      = 'Dish Name'

def _safe_float(val):
    try:
        v = float(val)
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0

def get_packaged_star(row):
    """Compute FSSAI solid-food star for a packaged row."""
    _, star = calculate_fssai_solid_score(
        _safe_float(row.get(PACKAGED_ENERGY_COL, 0)),
        _safe_float(row.get(PACKAGED_SAT_FAT_COL, 0)),
        _safe_float(row.get(PACKAGED_SUGAR_COL, 0)),
        _safe_float(row.get(PACKAGED_SODIUM_COL, 0)),
        _safe_float(row.get(PACKAGED_PROTEIN_COL, 0)),
        _safe_float(row.get(PACKAGED_FIBER_COL, 0)),
    )
    return star

def get_cooked_star(row):
    """Compute FSSAI solid-food star for a cooked row."""
    _, star = calculate_fssai_solid_score(
        _safe_float(row.get(COOKED_ENERGY_COL, 0)),
        _safe_float(row.get(COOKED_SAT_FAT_COL, 0)),
        _safe_float(row.get(COOKED_SUGAR_COL, 0)),
        _safe_float(row.get(COOKED_SODIUM_COL, 0)),
        _safe_float(row.get(COOKED_PROTEIN_COL, 0)),
        _safe_float(row.get(COOKED_FIBER_COL, 0)),
    )
    return star

def show_packaged_substitution_panel(current_star, current_name, df):
    """Renders healthier packaged alternatives if current rating <= 1.5."""
    if current_star > 1.5 or df.empty or PACKAGED_NAME_COL not in df.columns:
        return
    st.markdown("---")
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);'
        'border:1px solid #00FFCC55;border-radius:14px;padding:18px 22px;margin-bottom:16px;">'
        '<h3 style="color:#00FFCC;margin:0 0 6px 0;">💡 Try a Healthier Alternative!</h3>'
        '<p style="color:#b0b8c1;margin:0;font-size:0.93rem;">'
        'These items from our database score <strong style="color:#00FFCC;">3.5★ or higher</strong> '
        'and are significantly better nutritional choices.</p></div>',
        unsafe_allow_html=True
    )
    candidates = []
    for _, row in df.iterrows():
        name = str(row.get(PACKAGED_NAME_COL, ""))
        if name == current_name:
            continue
        star = get_packaged_star(row)
        if star >= 3.5:
            candidates.append((name, star,
                _safe_float(row.get(PACKAGED_ENERGY_COL, 0)),
                _safe_float(row.get(PACKAGED_PROTEIN_COL, 0)),
                _safe_float(row.get(PACKAGED_FIBER_COL, 0))))
        if len(candidates) >= 5:
            break
    if not candidates:
        st.info("No high-rated alternatives found in the current dataset subset.")
        return
    for name, star, energy, protein, fiber in candidates:
        full_s  = int(star)
        half_s  = 1 if star - full_s >= 0.5 else 0
        empty_s = 5 - full_s - half_s
        stars_d = "★" * full_s + ("⯨" if half_s else "") + "☆" * empty_s
        st.markdown(
            f'<div class="alt-card">'
            f'<h4>{name}</h4>'
            f'<span>{stars_d} &nbsp; {star}★ &nbsp;|&nbsp; '
            f'Energy: {energy} kcal &nbsp;|&nbsp; '
            f'Protein: {protein}g &nbsp;|&nbsp; Fiber: {fiber}g</span>'
            f'</div>',
            unsafe_allow_html=True
        )

def show_cooked_substitution_panel(current_star, current_name, df):
    """Renders healthier cooked alternatives if current rating <= 1.5."""
    if current_star > 1.5 or df.empty:
        return
    dish_col = COOKED_NAME_COL if COOKED_NAME_COL in df.columns else 'recipe_name'
    st.markdown("---")
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);'
        'border:1px solid #00FFCC55;border-radius:14px;padding:18px 22px;margin-bottom:16px;">'
        '<h3 style="color:#00FFCC;margin:0 0 6px 0;">💡 Try a Healthier Dish!</h3>'
        '<p style="color:#b0b8c1;margin:0;font-size:0.93rem;">'
        'These traditional dishes score <strong style="color:#00FFCC;">3.5★ or higher</strong>.</p></div>',
        unsafe_allow_html=True
    )
    candidates = []
    for _, row in df.iterrows():
        name = str(row.get(dish_col, ""))
        if name == current_name:
            continue
        star = get_cooked_star(row)
        if star >= 3.5:
            candidates.append((name, star,
                _safe_float(row.get(COOKED_ENERGY_COL, 0)),
                _safe_float(row.get(COOKED_PROTEIN_COL, 0)),
                _safe_float(row.get(COOKED_FIBER_COL, 0))))
        if len(candidates) >= 5:
            break
    if not candidates:
        st.info("No high-rated alternatives found in the current dataset subset.")
        return
    for name, star, energy, protein, fiber in candidates:
        full_s  = int(star)
        half_s  = 1 if star - full_s >= 0.5 else 0
        empty_s = 5 - full_s - half_s
        stars_d = "★" * full_s + ("⯨" if half_s else "") + "☆" * empty_s
        st.markdown(
            f'<div class="alt-card">'
            f'<h4>{name}</h4>'
            f'<span>{stars_d} &nbsp; {star}★ &nbsp;|&nbsp; '
            f'Energy: {energy} kcal &nbsp;|&nbsp; '
            f'Protein: {protein}g &nbsp;|&nbsp; Fiber: {fiber}g</span>'
            f'</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────
# MODULE 3 — PDF EXPORT ENGINE
# ─────────────────────────────────────────────────────────────

# Mapping of all special/emoji chars used in the app to ASCII-safe PDF equivalents
_PDF_CHAR_MAP = {
    "★": "*",  "☆": "o",  "⯨": ".5",  "½": ".5",
    "⚡": "",  "🧈": "",  "🍬": "",  "🧂": "",  "🥩": "",  "🌾": "",
    "⚠": "!!", "⚠️": "!!",
    "🟢": "",  "🟡": "",  "🔴": "",  "🚨": "",
    "📐": "",  "🥤": "",  "🤖": "",
    "💡": "",  "📥": "",  "📸": "",  "🎛️": "",  "🔍": "",
    "&": "and", ">": ">", "<": "<",
}

def _pdf_safe(text: str) -> str:
    """Strip / replace all characters outside Latin-1 so Helvetica never errors."""
    text = str(text)
    for char, replacement in _PDF_CHAR_MAP.items():
        text = text.replace(char, replacement)
    # Final pass: drop anything still outside latin-1 range
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def export_to_pdf(product_name, form, metrics_dict, final_score, star_rating,
                  hfss_warning=False):
    """
    Generates a structured FSSAI nutritional report PDF using fpdf2.
    Returns bytes of the PDF file, or None if fpdf2 is not installed.
    """
    # Sanitize typography characters that Helvetica cannot encode
    def _sanitize(s):
        return (str(s)
                .replace("\u2014", "-")   # em-dash —
                .replace("\u2013", "-")   # en-dash –
                .replace("\u2018", "'")
                .replace("\u2019", "'")
                .replace("\u201c", '"')
                .replace("\u201d", '"'))
    product_name = _sanitize(product_name)
    form         = _sanitize(form)

    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ──
    pdf.set_fill_color(23, 32, 42)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 255, 204)
    pdf.cell(0, 14, "FSSAI Indian Health Star Rating Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(180, 180, 180)
    pdf.cell(0, 8, "Generated by Indian Health Star Rating AI System", ln=True, align="C")
    pdf.ln(8)

    # ── Product details ──
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, _pdf_safe(f"Product / Dish: {product_name}"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _pdf_safe(f"Product Form:  {form}"), ln=True)
    pdf.ln(4)

    # ── HFSS Warning ──
    if hfss_warning:
        pdf.set_fill_color(248, 196, 113)
        pdf.set_text_color(147, 81, 22)
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 8,
            "!! HFSS Warning: High Fat, Sugar & Salt detected (Sodium-to-Energy ratio > 1.0)",
            fill=True
        )
        pdf.ln(3)

    # ── Nutritional matrix table ──
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Nutritional Breakdown (per 100g / 100ml):", ln=True)
    pdf.ln(2)

    headers  = ["Nutrient Component", "Value"]
    col_w    = [120, 60]
    row_h    = 8

    # Table header row
    pdf.set_fill_color(23, 32, 42)
    pdf.set_text_color(0, 255, 204)
    pdf.set_font("Helvetica", "B", 10)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], row_h, h, border=1, fill=True, align="C")
    pdf.ln()

    # Table data rows
    row_colors = [(245, 247, 250), (255, 255, 255)]
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 10)
    for idx, (nutrient, value) in enumerate(metrics_dict.items()):
        r, g, b = row_colors[idx % 2]
        pdf.set_fill_color(r, g, b)
        pdf.cell(col_w[0], row_h, _pdf_safe(nutrient), border=1, fill=True)
        pdf.cell(col_w[1], row_h, _pdf_safe(str(value)), border=1, fill=True, align="C")
        pdf.ln()

    pdf.ln(6)

    # ── Star rating banner ──
    pdf.set_fill_color(23, 32, 42)
    pdf.set_text_color(241, 196, 15)
    pdf.set_font("Helvetica", "B", 16)
    full_s = int(star_rating)
    half_s = 1 if star_rating - full_s >= 0.5 else 0
    empty_s = 5 - full_s - half_s
    stars_str = ("*" * full_s) + (".5" if half_s else "") + ("o" * empty_s)
    pdf.cell(0, 12, f"FSSAI Health Star Rating: {stars_str}  ({star_rating} / 5.0)",
             ln=True, fill=True, align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, f"Final FSSAI Score (lower = healthier): {final_score}", ln=True)

    # -- Tier label --
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    if star_rating >= 4.0:
        pdf.set_text_color(39, 174, 96)
        tier = "Excellent Choice - Highly Nutritious"
    elif star_rating >= 2.5:
        pdf.set_text_color(230, 126, 34)
        tier = "Average - Consume in Moderation"
    elif star_rating >= 1.5:
        pdf.set_text_color(192, 57, 43)
        tier = "Poor Nutritional Value - Limit Intake"
    else:
        pdf.set_text_color(169, 50, 38)
        tier = "Very Poor - High Health Risk"
    pdf.cell(0, 8, _pdf_safe(tier), ln=True)

    # ── Footer ──
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 6,
        "This report is based on FSSAI Indian Nutrition Rating (INR) guidelines. "
        "For professional dietary advice, consult a certified nutritionist.",
        ln=True, align="C"
    )

    return bytes(pdf.output())

def render_pdf_download_button(product_name, form, metrics_dict,
                                final_score, star_rating, hfss_warning=False):
    """Renders the st.download_button for the PDF export."""
    pdf_bytes = export_to_pdf(
        product_name, form, metrics_dict, final_score, star_rating, hfss_warning
    )
    if pdf_bytes is None:
        st.warning("📦 PDF export requires `fpdf2`. Install it with `pip install fpdf2`.")
        return
    safe_name = re.sub(r'[^\w\s-]', '', product_name).strip().replace(' ', '_')
    st.download_button(
        label="📥 Export Nutritional Report PDF",
        data=pdf_bytes,
        file_name=f"{safe_name}_FSSAI_Report.pdf",
        mime="application/pdf",
        use_container_width=False
    )


# ─────────────────────────────────────────────────────────────
# ANIMATION HTML — RAF-optimised cursor eye-tracking
# ─────────────────────────────────────────────────────────────

def _build_eye_tracker_html(emojis_with_eyes, height=160):
    """
    Builds the full HTML/CSS/JS block for the floating emoji eye-tracker.
    emojis_with_eyes: list of tuples (emoji_char, left_eye_offset, right_eye_offset)
                      where offsets are (top_px, left_px) tuples.
    Uses requestAnimationFrame for lag-free rendering.
    """
    float_anims = "\n".join([
        "@keyframes float1 { 0% { transform: translate(10px,20px) rotate(-5deg); } 100% { transform: translate(180px,60px) rotate(15deg); } }",
        "@keyframes float2 { 0% { transform: translate(200px,10px) rotate(10deg); } 100% { transform: translate(20px,80px) rotate(-10deg); } }",
        "@keyframes float3 { 0% { transform: translate(80px,80px) rotate(-15deg); } 100% { transform: translate(160px,10px) rotate(5deg); } }",
        "@keyframes float4 { 0% { transform: translate(150px,90px) rotate(5deg); } 100% { transform: translate(50px,20px) rotate(-5deg); } }",
        "@keyframes float5 { 0% { transform: translate(120px,10px) rotate(0deg); } 100% { transform: translate(100px,90px) rotate(10deg); } }",
    ])
    durations = [12, 9, 14, 11, 15]
    face_divs = ""
    for i, (emoji, (lt, ll), (rt, rl)) in enumerate(emojis_with_eyes, start=1):
        face_divs += (
            f'<div class="face" id="e{i}">{emoji}'
            f'<div class="eye" style="top:{lt}px;left:{ll}px;"><div class="pupil"></div></div>'
            f'<div class="eye" style="top:{rt}px;left:{rl}px;"><div class="pupil"></div></div>'
            f'</div>\n'
        )
    id_rules = "\n".join(
        [f"#e{i} {{ animation-name: float{i}; animation-duration: {durations[i-1]}s; top:0; left:0; }}"
         for i in range(1, len(emojis_with_eyes) + 1)]
    )
    return f"""
    <style>
    body {{ margin:0; padding:0; display:block; height:{height}px; font-family:sans-serif;
            background:transparent; overflow:hidden; position:relative; }}
    .face {{ position:absolute; font-size:4.2rem; animation-timing-function:ease-in-out;
             animation-iteration-count:infinite; animation-direction:alternate; }}
    .eye {{ position:absolute; width:14px; height:14px; background:white; border-radius:50%;
            border:1.5px solid black; display:flex; align-items:center;
            justify-content:flex-end; transition:height 0.1s; }}
    .pupil {{ width:5px; height:5px; background:black; border-radius:50%;
              margin-right:1px; transition:transform 0.1s; }}
    .blink .eye {{ height:2px; margin-top:6px; background:black; border:none; }}
    .blink .pupil {{ display:none; }}
    {float_anims}
    {id_rules}
    .paused {{ animation-play-state:paused !important; }}
    </style>
    <div id="pool">{face_divs}</div>
    <script>
    let idleTimer;
    let isIdle = true;
    const eyes  = document.querySelectorAll('.eye');
    const faces = document.querySelectorAll('.face');

    const resetIdleTimer = () => {{
      if (isIdle) {{
        isIdle = false;
        faces.forEach(f => f.classList.add('paused'));
      }}
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {{
        isIdle = true;
        faces.forEach(f => f.classList.remove('paused'));
        eyes.forEach(eye => {{ eye.style.transform = 'rotate(0deg)'; }});
      }}, 3000);
    }};

    const triggerBlink = () => {{
      if (!isIdle) return;
      faces.forEach(f => f.classList.add('blink'));
      setTimeout(() => {{
        faces.forEach(f => f.classList.remove('blink'));
        setTimeout(() => {{
          faces.forEach(f => f.classList.add('blink'));
          setTimeout(() => {{ faces.forEach(f => f.classList.remove('blink')); }}, 100);
        }}, 100);
      }}, 100);
    }};
    setInterval(triggerBlink, 4000);

    let mouseX = 0, mouseY = 0, ticking = false;

    const updateEyes = () => {{
      eyes.forEach(eye => {{
        const rect = eye.getBoundingClientRect();
        const cx   = rect.left + rect.width  / 2;
        const cy   = rect.top  + rect.height / 2;
        const rad  = Math.atan2(mouseY - cy, mouseX - cx);
        eye.style.transform = `rotate(${{rad * 180 / Math.PI}}deg)`;
      }});
    }};

    const onMouseMove = (e) => {{
      resetIdleTimer();
      let tx = e.clientX, ty = e.clientY;
      if (e.view === window.parent && window.frameElement) {{
        const ir = window.frameElement.getBoundingClientRect();
        tx -= ir.left; ty -= ir.top;
      }}
      mouseX = tx; mouseY = ty;
      if (!ticking) {{
        window.requestAnimationFrame(() => {{ updateEyes(); ticking = false; }});
        ticking = true;
      }}
    }};

    window.addEventListener('mousemove', onMouseMove);
    if (window.parent) window.parent.addEventListener('mousemove', onMouseMove);
    resetIdleTimer();
    </script>
    """


# ─────────────────────────────────────────────────────────────
# SHARED CSS INJECTION — NEON BUTTONS & SELECTBOXES
# ─────────────────────────────────────────────────────────────

NEON_CSS = """
<style>
.stButton > button {
    background-color: #1E1E1E !important;
    color: white !important;
    border: 2px solid #333 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    padding: 18px 30px !important;
}
.stButton > button:hover {
    border-color: #00FFCC !important;
    box-shadow: 0 0 15px #00FFCC, 0 0 30px #00FFCC !important;
    transform: scale(1.02) !important;
    color: #00FFCC !important;
}
div[data-testid="stSelectbox"] {
    max-width: 400px !important;
    margin-left: 0 !important;
}
div[data-baseweb="select"] {
    transition: all 0.3s ease !important;
}
div[data-baseweb="select"]:hover,
div[data-baseweb="select"]:focus-within {
    border-color: #00FFCC !important;
    box-shadow: 0 0 15px #00FFCC, 0 0 30px #00FFCC !important;
}
div[data-testid="stNumberInput"] {
    max-width: 320px !important;
}
</style>
"""

NUTRIENT_LABELS = ["⚡ Energy", "🧈 Saturated Fat", "🍬 Total Sugars",
                   "🧂 Sodium", "🥩 Protein", "🌾 Dietary Fiber"]

def build_metrics_dict(energy, fat, sat_fat, sugar, sodium, protein, fiber, form):
    unit_e = "kcal" if form == "Solid Food" else "kcal/100ml"
    return {
        "⚡ Energy":         f"{energy} {unit_e}",
        "🧈 Saturated Fat":  f"{fat} g",
        "🍬 Total Sugars":   f"{sugar} g",
        "🧂 Sodium":         f"{sodium} mg",
        "🥩 Protein":        f"{protein} g",
        "🌾 Dietary Fiber":  f"{fiber} g",
    }


# ─────────────────────────────────────────────────────────────
# SESSION STATE BOOTSTRAPPING
# ─────────────────────────────────────────────────────────────

if 'active_mode' not in st.session_state:
    st.session_state.active_mode = "landing"

def set_mode(mode):
    st.session_state.active_mode = mode
    if mode != "packaged":
        st.session_state.packaged_sub_mode = None


# ═══════════════════════════════════════════════════════════════
# PAGE: LANDING
# ═══════════════════════════════════════════════════════════════

if st.session_state.active_mode == "landing":
    st.markdown("""
    <style>
    .stButton > button {
        background-color: #1E1E1E !important;
        color: white !important;
        border: 2px solid #333 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        border-color: #00FFCC !important;
        box-shadow: 0 0 15px #00FFCC, 0 0 30px #00FFCC !important;
        transform: scale(1.02) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🌟 FSSAI Indian Health Star Rating System</div>',
                unsafe_allow_html=True)

    _, card_left, card_right, _ = st.columns([1, 4, 4, 1])

    # ── Left card: Packaged Foods ──
    with card_left:
        packaged_emojis = [
            ("🥤", (24, 12), (24, 42)),
            ("🍿", (30, 14), (30, 44)),
            ("🍪", (30, 18), (30, 48)),
            ("🍫", (24, 12), (24, 42)),
            ("🧃", (36, 14), (36, 44)),
        ]
        html_left = _build_eye_tracker_html(packaged_emojis, height=160)
        st.components.v1.html(html_left, height=160)
        st.subheader("📦 1. Packaged Foods Classifier")
        st.markdown(
            '<div style="height:80px;">Scan a Nutrition Information Panel (NIP) using '
            'Vision AI to automatically extract and score commercial food products.</div>',
            unsafe_allow_html=True
        )
        st.button("Proceed to Packaged Foods", on_click=set_mode,
                  args=("packaged",), use_container_width=True)

    # ── Right card: Cooked Dishes ──
    with card_right:
        cooked_emojis = [
            ("🍲", (30, 10), (30, 40)),
            ("☕", (30, 18), (30, 48)),
            ("🍚", (30, 18), (30, 48)),
            ("🥟", (24, 18), (24, 48)),
            ("🍢", (24, 18), (24, 42)),
        ]
        html_right = _build_eye_tracker_html(cooked_emojis, height=160)
        st.components.v1.html(html_right, height=160)
        st.subheader("🍲 2. Traditional Cooked Dishes")
        st.markdown(
            '<div style="height:80px;">Search and evaluate traditional Indian recipes '
            'against strict FSSAI nutritional rating guidelines.</div>',
            unsafe_allow_html=True
        )
        st.button("Proceed to Cooked Dishes", on_click=set_mode,
                  args=("cooked",), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: PACKAGED FOODS
# ═══════════════════════════════════════════════════════════════

elif st.session_state.active_mode == "packaged":
    st.button("⬅️ Back to Home", on_click=set_mode, args=("landing",))
    model_key = "Packaged Foods"
    st.markdown("### 🛒 Packaged Foods Scanner")

    # Initialise sub-mode to clean slate
    if 'packaged_sub_mode' not in st.session_state:
        st.session_state.packaged_sub_mode = None

    def set_sub_mode(mode):
        st.session_state.packaged_sub_mode = mode

    st.markdown(NEON_CSS, unsafe_allow_html=True)

    # ── Vertically stacked, centered sub-navigation ──
    _, center_col, _ = st.columns([1, 1, 1])
    with center_col:
        st.button("🔍 1. Search food",
                  on_click=set_sub_mode, args=("Search Product",),
                  use_container_width=True)
        st.button("📸 2. Upload NIP",
                  on_click=set_sub_mode, args=("Upload NIP",),
                  use_container_width=True)
        st.button("🎛️ 3. Manually type NIP.",
                  on_click=set_sub_mode, args=("Manually Typing NIP",),
                  use_container_width=True)

    st.markdown("---")

    # ── Sub-workspace: Search Food ──
    if st.session_state.packaged_sub_mode == "Search Product":
        st.markdown("#### 🔍 Search Food")
        st.markdown("Search the dataset to look up nutritional parameters.")
        df = load_dataset()

        search_col1, search_col2, _ = st.columns([2, 2, 1], gap="small")
        selected_product = None
        with search_col1:
            if not df.empty and PACKAGED_NAME_COL in df.columns:
                selected_product = st.selectbox(
                    "Search registered commercial food items:",
                    df[PACKAGED_NAME_COL].unique(),
                    key="pkg_search_item"
                )
        with search_col2:
            search_product_form = st.selectbox(
                "Product Form:",
                ["Solid Food", "Liquid Beverage"],
                key="search_product_form_selection"
            )

        if selected_product and not df.empty and PACKAGED_NAME_COL in df.columns:
            row = df[df[PACKAGED_NAME_COL] == selected_product].iloc[0]
            energy_val  = _safe_float(row.get(PACKAGED_ENERGY_COL,  0))
            fat_val     = _safe_float(row.get(PACKAGED_FAT_COL,     0))
            sat_fat_val = _safe_float(row.get(PACKAGED_SAT_FAT_COL, 0))
            sugar_val   = _safe_float(row.get(PACKAGED_SUGAR_COL,   0))
            sodium_val  = _safe_float(row.get(PACKAGED_SODIUM_COL,  0))
            protein_val = _safe_float(row.get(PACKAGED_PROTEIN_COL, 0))
            fiber_val   = _safe_float(row.get(PACKAGED_FIBER_COL,   0))

            st.markdown(f"**{selected_product}** Nutritional Breakdown:")
            metrics = build_metrics_dict(energy_val, fat_val, sat_fat_val, sugar_val,
                                         sodium_val, protein_val, fiber_val,
                                         search_product_form)
            table_data = {
                "Nutrient Component": list(metrics.keys()),
                "Value per 100g/ml":  list(metrics.values()),
            }
            st.table(pd.DataFrame(table_data))

            hfss = energy_val > 0 and (sodium_val / energy_val) > 1.0
            final_score, star = display_fssai_rating(
                energy_val, sat_fat_val, sugar_val, sodium_val, protein_val, fiber_val,
                product_category=search_product_form,
                product_name=selected_product
            )

            render_pdf_download_button(
                selected_product, search_product_form, metrics,
                final_score, star, hfss_warning=hfss
            )
            show_packaged_substitution_panel(star, selected_product, df)

        elif df.empty or PACKAGED_NAME_COL not in df.columns:
            st.warning("Dataset not found or missing 'Item name' column.")

    # ── Sub-workspace: Upload NIP (Vision AI) ──
    elif st.session_state.packaged_sub_mode == "Upload NIP":
        st.markdown("#### 📸 Vision AI Label Scanner")
        product_category = st.selectbox(
            "Product Form:", ["Solid Food", "Liquid Beverage"],
            key="vision_product_form"
        )
        st.markdown("Upload a Nutrition Information Panel (NIP) image to automatically "
                    "extract metrics using Vision AI OCR.")
        uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img_col, _ = st.columns([1, 2])
            with img_col:
                st.image(img, caption='Uploaded NIP Image', width=350)

            # ── 3D Scanning Animation placeholder ──
            anim_slot = st.empty()
            anim_slot.components_html = None
            anim_slot.markdown("""
            <style>
            @keyframes radarSpin {
                0%   { transform: rotateY(0deg) rotateX(15deg); }
                100% { transform: rotateY(360deg) rotateX(15deg); }
            }
            @keyframes scanLine {
                0%   { top: 10%; opacity: 1; }
                80%  { top: 90%; opacity: 1; }
                100% { top: 10%; opacity: 0.3; }
            }
            @keyframes pulse3d {
                0%, 100% { box-shadow: 0 0 18px #00FFCC, 0 0 40px #00FFCC44; }
                50%       { box-shadow: 0 0 40px #00FFCC, 0 0 80px #00FFCC88; }
            }
            .ocr-scene {
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                padding: 24px 0 10px 0;
                background: linear-gradient(135deg, #0a0f1e 0%, #0d1a2e 100%);
                border-radius: 16px;
                border: 1px solid #00FFCC33;
                margin-bottom: 16px;
                animation: pulse3d 2s ease-in-out infinite;
            }
            .ocr-ring-wrap {
                perspective: 600px;
                width: 120px; height: 120px;
                display: flex; align-items: center; justify-content: center;
            }
            .ocr-ring {
                width: 110px; height: 110px;
                border: 4px solid transparent;
                border-top-color: #00FFCC;
                border-right-color: #00FFCC88;
                border-radius: 50%;
                animation: radarSpin 1.4s linear infinite;
                position: relative;
                transform-style: preserve-3d;
            }
            .ocr-ring::before {
                content: '';
                position: absolute;
                inset: 12px;
                border: 3px solid transparent;
                border-bottom-color: #00FFCC;
                border-left-color: #00FFCC66;
                border-radius: 50%;
                animation: radarSpin 0.9s linear infinite reverse;
            }
            .ocr-ring::after {
                content: '🔍';
                position: absolute;
                top: 50%; left: 50%;
                transform: translate(-50%, -50%);
                font-size: 2rem;
            }
            .ocr-scan-bar-wrap {
                position: relative;
                width: 220px; height: 6px;
                background: #1a2a3a;
                border-radius: 3px;
                overflow: hidden;
                margin: 18px 0 10px 0;
            }
            .ocr-scan-bar {
                height: 100%;
                width: 40%;
                background: linear-gradient(90deg, transparent, #00FFCC, transparent);
                border-radius: 3px;
                animation: scanLine 1.6s ease-in-out infinite;
                position: absolute;
                left: 0;
            }
            .ocr-label {
                color: #00FFCC;
                font-family: 'Inter', monospace;
                font-size: 0.95rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-align: center;
            }
            .ocr-sublabel {
                color: #5a8a7a;
                font-size: 0.75rem;
                margin-top: 4px;
                letter-spacing: 0.05em;
            }
            </style>
            <div class="ocr-scene">
                <div class="ocr-ring-wrap">
                    <div class="ocr-ring"></div>
                </div>
                <div class="ocr-scan-bar-wrap"><div class="ocr-scan-bar"></div></div>
                <div class="ocr-label">⚡ VISION AI SCANNING…</div>
                <div class="ocr-sublabel">Extracting nutritional data from NIP image</div>
            </div>
            """, unsafe_allow_html=True)

            try:
                results        = reader.readtext(np.array(img), detail=0)
                extracted_text = " ".join(results)
            except Exception as e:
                anim_slot.empty()
                st.error(f"Vision AI OCR failed: {e}")
                extracted_text = ""

            anim_slot.empty()   # remove animation once OCR completes

            if not extracted_text.strip():
                st.error("❌ Could not detect any text. "
                         "Please ensure the image clearly shows the nutritional values table.")
            else:
                if not validate_nip_content(extracted_text):
                    st.error("❌ Could not detect a valid NIP. "
                             "Please ensure the image clearly shows the nutritional values table.")
                else:
                    st.text_area("Extracted Raw Text", extracted_text, height=120)
                    with st.spinner("Parsing nutritional parameters via Regex..."):
                        parsed_data = parse_nutritional_info(extracted_text)

                    # ── Partial-extraction guard ──
                    missing_fields = [k for k, v in parsed_data.items() if v == 0.0]
                    if len(missing_fields) >= 4:
                        st.warning(
                            "⚠️ **Partial extraction detected.** "
                            f"The following fields could not be read from the image and have been "
                            f"set to 0: **{', '.join(missing_fields)}**. "
                            "Please fill them in manually below before calculating."
                        )
                    elif missing_fields:
                        st.info(
                            f"ℹ️ Some fields were not detected (**{', '.join(missing_fields)}**) "
                            "and defaulted to 0. Verify and adjust below if needed."
                        )

                    st.subheader("✏️ Verify & Adjust Extracted Values")
                    c1, c2 = st.columns(2)
                    with c1:
                        energy_val  = st.number_input("Energy (kcal per 100g)",
                            min_value=0.0,
                            value=float(parsed_data.get('Energy',  0.0)),
                            key="ocr_en",
                            help="Required for scoring" if parsed_data.get('Energy', 0.0) == 0.0 else None)
                        fat_val     = st.number_input("Saturated Fat (g per 100g)",
                            min_value=0.0,
                            value=float(parsed_data.get('Sat_Fat', 0.0)),
                            key="ocr_fat")
                        sugar_val   = st.number_input("Total Sugars (g per 100g)",
                            min_value=0.0,
                            value=float(parsed_data.get('Sugar',   0.0)),
                            key="ocr_sug")
                    with c2:
                        sodium_val  = st.number_input("Sodium (mg per 100g)",
                            min_value=0.0,
                            value=float(parsed_data.get('Sodium',  0.0)),
                            key="ocr_sod")
                        protein_val = st.number_input("Protein (g per 100g)",
                            min_value=0.0,
                            value=float(parsed_data.get('Protein', 0.0)),
                            key="ocr_pro")
                        fiber_val   = st.number_input("Dietary Fiber (g per 100g)",
                            min_value=0.0,
                            value=float(parsed_data.get('Fiber',   0.0)),
                            key="ocr_fib")

                    st.markdown("---")

                    # ── Minimum completeness check before enabling Calculate ──
                    if energy_val == 0.0:
                        st.error(
                            "🚫 **Cannot calculate:** Energy (kcal) is required and is currently 0. "
                            "Please enter the energy value from the NIP label above."
                        )
                    else:
                        if st.button("⚡ Calculate FSSAI Rating",
                                     key="ocr_calc_btn",
                                     use_container_width=False):
                            display_ml_prediction(
                                energy_val, sat_fat_val, sugar_val, sodium_val,
                                protein_val, fiber_val,
                                model_key=model_key,
                                product_category=product_category
                            )
                            metrics = build_metrics_dict(
                                energy_val, fat_val, sugar_val,
                                sodium_val, protein_val, fiber_val,
                                product_category
                            )
                            _, star = calculate_fssai_rating(
                                energy_val, fat_val, sugar_val,
                                sodium_val, protein_val, fiber_val,
                                product_category
                            )
                            hfss = sodium_val / energy_val > 1.0
                            render_pdf_download_button(
                                uploaded_file.name, product_category, metrics,
                                _, star, hfss_warning=hfss
                            )

    # ── Sub-workspace: Manually Type NIP ──
    elif st.session_state.packaged_sub_mode == "Manually Typing NIP":
        st.markdown("#### 🎛️ Enter NIP Manually")
        st.markdown("Enter numerical parameters (per 100g) below, then click **Calculate**.")

        man_col1, _ = st.columns([1, 1])
        with man_col1:
            manual_product_form = st.selectbox(
                "Product Form:",
                ["Solid Food", "Liquid Beverage"],
                key="manual_product_form_selection"
            )

        col1, col2 = st.columns(2)
        with col1:
            energy_man  = st.number_input("Energy / Calories (kcal)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0,  key="pkg_en")
            fat_man     = st.number_input("Total Fat (g)",         min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="pkg_tot_fat")
            sat_fat_man = st.number_input("Saturated Fat (g)",     min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="pkg_sat_fat")
            sugar_man   = st.number_input("Sugar (g)",                 min_value=0.0, max_value=100.0,  value=0.0, step=0.1,  key="pkg_sug")
        with col2:
            sodium_man  = st.number_input("Sodium (mg)",               min_value=0.0, max_value=5000.0, value=0.0, step=1.0,  key="pkg_sod")
            protein_man = st.number_input("Protein (g)",               min_value=0.0, max_value=100.0,  value=0.0, step=0.1,  key="pkg_pro")
            fiber_man   = st.number_input("Dietary Fiber (g)",         min_value=0.0, max_value=100.0,  value=0.0, step=0.1,  key="pkg_fib")

        # ── Minimum completeness check ──
        filled_count = sum(1 for v in [energy_man, fat_man, sat_fat_man, sugar_man, sodium_man, protein_man, fiber_man] if v > 0.0)
        if energy_man == 0.0:
            st.error(
                "🚫 **Energy (kcal) is required.** "
                "Please enter at least the Energy value before calculating."
            )
        elif filled_count < 3:
            st.warning(
                f"⚠️ Only **{filled_count} of 6** fields have values above zero. "
                "For an accurate FSSAI rating, please fill in as many fields as possible. "
                "Click Calculate to proceed with available values."
            )

        st.markdown("---")
        if st.button("⚡ Calculate FSSAI Rating",
                     key="manual_calc_btn",
                     use_container_width=False):
            if energy_man == 0.0:
                st.error("🚫 Cannot calculate: Energy value is 0. Please enter a valid Energy reading.")
            else:
                hfss = sodium_man > 0 and (sodium_man / energy_man) > 1.0
                final_score, star = display_fssai_rating(
                    energy_man, sat_fat_man, sugar_man, sodium_man, protein_man, fiber_man,
                    product_category=manual_product_form
                )
                metrics = build_metrics_dict(energy_man, fat_man, sat_fat_man, sugar_man,
                                             sodium_man, protein_man, fiber_man,
                                             manual_product_form)
                render_pdf_download_button(
                    "Manual NIP Entry", manual_product_form, metrics,
                    final_score, star, hfss_warning=hfss
                )


# ═══════════════════════════════════════════════════════════════
# PAGE: TRADITIONAL COOKED DISHES
# ═══════════════════════════════════════════════════════════════

elif st.session_state.active_mode == "cooked":
    st.button("⬅️ Back to Home", on_click=set_mode, args=("landing",))
    st.markdown("#### 🔍 Search Traditional Dishes")
    st.markdown(NEON_CSS, unsafe_allow_html=True)

    df_cooked  = load_cooked_dataset()
    dish_column = COOKED_NAME_COL if COOKED_NAME_COL in df_cooked.columns else 'recipe_name'

    if not df_cooked.empty and dish_column in df_cooked.columns:
        col_dish, col_form = st.columns(2)
        with col_dish:
            selected_dish = st.selectbox(
                "Search registered traditional Indian dishes:",
                df_cooked[dish_column].unique(),
                key="cooked_dish_selector"
            )
        with col_form:
            cooked_product_form = st.selectbox(
                "Product Form:",
                ["Solid Food", "Liquid Beverage"],
                key="cooked_product_form_selection"
            )

        if selected_dish:
            row = df_cooked[df_cooked[dish_column] == selected_dish].iloc[0]

            energy_val  = _safe_float(row.get(COOKED_ENERGY_COL,   0))
            fat_val     = _safe_float(row.get(COOKED_FAT_COL,      0))
            sat_fat_val = _safe_float(row.get(COOKED_SAT_FAT_COL, 0))
            sugar_val   = _safe_float(row.get(COOKED_SUGAR_COL,    0))
            sodium_val  = _safe_float(row.get(COOKED_SODIUM_COL,   0))
            protein_val = _safe_float(row.get(COOKED_PROTEIN_COL,  0))
            fiber_val   = _safe_float(row.get(COOKED_FIBER_COL,    0))

            st.markdown(f"**{selected_dish}** Nutritional Breakdown:")
            metrics = build_metrics_dict(energy_val, fat_val, sat_fat_val, sugar_val,
                                         sodium_val, protein_val, fiber_val,
                                         cooked_product_form)
            table_data = {
                "Nutrient Component": list(metrics.keys()),
                "Value per 100g/ml":  list(metrics.values()),
            }
            st.table(pd.DataFrame(table_data))

            hfss = energy_val > 0 and (sodium_val / energy_val) > 1.0
            final_score, star = display_fssai_rating(
                energy_val, sat_fat_val, sugar_val, sodium_val, protein_val, fiber_val,
                product_category=cooked_product_form,
                product_name=selected_dish
            )

            # Papad serving-size context note
            if "papad" in selected_dish.lower():
                st.info(
                    "\U0001f4a1 **Serving Size Note:** Papad is evaluated per 100g raw weight "
                    "according to standard FSSAI protocols. While it scales with high sodium "
                    "density, a typical single-piece serving is only ~3g to 5g in practice."
                )

            render_pdf_download_button(
                selected_dish, cooked_product_form, metrics,
                final_score, star, hfss_warning=hfss
            )
            show_cooked_substitution_panel(star, selected_dish, df_cooked)

    else:
        st.warning("Dataset not found or missing dish name column.")
