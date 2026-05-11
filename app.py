# ============================================================
#  UHPFRC Compressive Strength Predictor  –  Streamlit App
#  Model : XGBoost  |  Target : CS (MPa)  |  Age fixed = 28d
# ============================================================

import streamlit as st
import pickle
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "UHPFRC CS Predictor",
    page_icon  = "🏗️",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  –  engineering-grade dark UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font import ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary   : #0D1117;
    --bg-card      : #161B22;
    --bg-input     : #1C2128;
    --accent       : #00D4AA;
    --accent-dim   : #00A882;
    --accent-glow  : rgba(0, 212, 170, 0.15);
    --danger       : #FF6B6B;
    --text-primary : #E6EDF3;
    --text-muted   : #7D8590;
    --border       : #30363D;
    --border-accent: #00D4AA;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family : 'DM Sans', sans-serif !important;
    background  : var(--bg-primary) !important;
    color       : var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background  : var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family : 'Syne', sans-serif !important;
    color       : var(--accent) !important;
    font-size   : 0.8rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* ── Number inputs ── */
input[type="number"] {
    background   : var(--bg-input)   !important;
    border       : 1px solid var(--border) !important;
    border-radius: 6px !important;
    color        : var(--text-primary) !important;
    font-family  : 'DM Mono', monospace !important;
    font-size    : 0.88rem !important;
    transition   : border-color 0.2s ease !important;
}
input[type="number"]:focus {
    border-color : var(--border-accent) !important;
    box-shadow   : 0 0 0 3px var(--accent-glow) !important;
}

/* ── Labels ── */
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background   : var(--accent) !important;
    color        : #0D1117 !important;
    font-family  : 'Syne', sans-serif !important;
    font-weight  : 800 !important;
    font-size    : 1rem !important;
    letter-spacing: 0.06em !important;
    border       : none !important;
    border-radius: 8px !important;
    padding      : 0.75rem 2.5rem !important;
    width        : 100% !important;
    transition   : all 0.2s ease !important;
    cursor       : pointer !important;
}
div[data-testid="stButton"] > button:hover {
    background  : var(--accent-dim) !important;
    transform   : translateY(-1px) !important;
    box-shadow  : 0 6px 20px var(--accent-glow) !important;
}

/* ── Result card ── */
.result-card {
    background     : linear-gradient(135deg, #0D2B24 0%, #0D1117 60%);
    border         : 1px solid var(--accent);
    border-radius  : 12px;
    padding        : 2rem 2.5rem;
    text-align     : center;
    box-shadow     : 0 0 40px var(--accent-glow);
    margin-top     : 1.5rem;
    animation      : fadeIn 0.4s ease;
}
.result-label {
    font-family    : 'DM Mono', monospace;
    font-size      : 0.72rem;
    letter-spacing : 0.18em;
    text-transform : uppercase;
    color          : var(--accent);
    margin-bottom  : 0.4rem;
}
.result-value {
    font-family : 'Syne', sans-serif;
    font-size   : 3.8rem;
    font-weight : 800;
    color       : var(--text-primary);
    line-height : 1;
}
.result-unit {
    font-family : 'DM Mono', monospace;
    font-size   : 1.1rem;
    color       : var(--text-muted);
    margin-top  : 0.3rem;
}
.strength-tag {
    display        : inline-block;
    margin-top     : 1rem;
    padding        : 0.3rem 1rem;
    border-radius  : 20px;
    font-family    : 'DM Mono', monospace;
    font-size      : 0.75rem;
    letter-spacing : 0.08em;
    font-weight    : 500;
}

/* ── Info / warning cards ── */
.info-card {
    background    : var(--bg-card);
    border        : 1px solid var(--border);
    border-left   : 3px solid var(--accent);
    border-radius : 8px;
    padding       : 0.9rem 1.2rem;
    font-size     : 0.82rem;
    color         : var(--text-muted);
    margin-bottom : 1rem;
}

/* ── Section divider ── */
.section-label {
    font-family    : 'DM Mono', monospace;
    font-size      : 0.68rem;
    letter-spacing : 0.18em;
    text-transform : uppercase;
    color          : var(--accent);
    border-bottom  : 1px solid var(--border);
    padding-bottom : 0.4rem;
    margin-bottom  : 0.8rem;
    margin-top     : 1.2rem;
}

/* ── Hero header ── */
.hero-title {
    font-family : 'Syne', sans-serif;
    font-size   : 2.6rem;
    font-weight : 800;
    line-height : 1.15;
    letter-spacing: -0.02em;
}
.hero-accent { color: var(--accent); }
.hero-sub {
    font-size  : 0.92rem;
    color      : var(--text-muted);
    margin-top : 0.5rem;
    max-width  : 520px;
}

/* ── Metric chip ── */
.chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.6rem; }
.chip {
    background     : var(--bg-card);
    border         : 1px solid var(--border);
    border-radius  : 20px;
    padding        : 0.25rem 0.8rem;
    font-family    : 'DM Mono', monospace;
    font-size      : 0.72rem;
    color          : var(--text-muted);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL  (cached so it only loads once per session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str = "xgboost_uhpfrc_model.pkl"):
    """Load the pickled XGBoost model from disk."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()


# ─────────────────────────────────────────────────────────────
# HELPER : classify strength grade
# ─────────────────────────────────────────────────────────────
def classify_strength(cs: float):
    """Return a (label, hex_color) tuple based on CS value (MPa)."""
    if cs >= 150:
        return "Ultra-High Strength  ≥ 150 MPa", "#00D4AA"
    elif cs >= 120:
        return "Very High Strength   120–150 MPa", "#4FC3F7"
    elif cs >= 100:
        return "High Strength        100–120 MPa", "#FFB74D"
    else:
        return "Moderate Strength    < 100 MPa", "#FF6B6B"


# ─────────────────────────────────────────────────────────────
# SIDEBAR  –  input controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔩 Mix Proportions")
    st.markdown('<div class="info-card">Quantities in kg/m³ unless noted.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-label">Binder Materials</div>',
                unsafe_allow_html=True)
    C  = st.number_input("C  – Cement",              min_value=0.0, value=800.0,  step=10.0, format="%.2f")
    SF = st.number_input("SF – Silica Fume",          min_value=0.0, value=150.0,  step=5.0,  format="%.2f")
    QP = st.number_input("QP – Quartz Powder",        min_value=0.0, value=0.0,    step=5.0,  format="%.2f")
    FA = st.number_input("FA – Fly Ash",              min_value=0.0, value=0.0,    step=5.0,  format="%.2f")
    SL = st.number_input("SL – Slag",                 min_value=0.0, value=0.0,    step=5.0,  format="%.2f")
    MK = st.number_input("MK – Metakaolin",           min_value=0.0, value=0.0,    step=5.0,  format="%.2f")

    st.markdown('<div class="section-label">Aggregates</div>',
                unsafe_allow_html=True)
    S  = st.number_input("S  – Sand",                 min_value=0.0, value=1100.0, step=10.0, format="%.2f")
    QS = st.number_input("QS – Quartz Sand",          min_value=0.0, value=0.0,    step=10.0, format="%.2f")

    st.markdown('<div class="section-label">Water & Admixtures</div>',
                unsafe_allow_html=True)
    W  = st.number_input("W  – Water",                min_value=0.0, value=180.0,  step=5.0,  format="%.2f")
    SP = st.number_input("SP – Superplasticizer",     min_value=0.0, value=20.0,   step=1.0,  format="%.2f")

    st.markdown('<div class="section-label">Fibre Parameters</div>',
                unsafe_allow_html=True)
    L   = st.number_input("L   – Fibre Length (mm)",  min_value=0.0, value=13.0,   step=0.5,  format="%.2f")
    D   = st.number_input("D   – Fibre Diameter (mm)",min_value=0.0, value=0.2,    step=0.01, format="%.3f")
    BV  = st.number_input("BV  – Brass Fibre Vol. (%)",min_value=0.0, value=0.0,  step=0.1,  format="%.2f")
    PPV = st.number_input("PPV – PP Fibre Vol. (%)",  min_value=0.0, value=0.0,    step=0.1,  format="%.2f")
    GV  = st.number_input("GV  – Glass Fibre Vol. (%)",min_value=0.0,value=0.0,   step=0.1,  format="%.2f")
    SSV = st.number_input("SSV – Steel Fibre Vol. (%)",min_value=0.0,value=2.0,   step=0.1,  format="%.2f")

    st.markdown("---")
    st.markdown(
        '<div class="info-card">⏱️ <b>Age fixed at 28 days</b><br>'
        'Standard curing period per IS 516 / ASTM C39.</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────
col_hero, col_badge = st.columns([3, 1])

with col_hero:
    st.markdown(
        '<div class="hero-title">'
        'UHPFRC <span class="hero-accent">Strength</span><br>Predictor'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-sub">'
        'XGBoost regression model trained on 864 experimental data points. '
        'Predicts 28-day compressive strength from mix design parameters.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="chip-row">'
        '<span class="chip">XGBoost v3.x</span>'
        '<span class="chip">864 samples</span>'
        '<span class="chip">17 features</span>'
        '<span class="chip">Age = 28 d</span>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Model status banner ───────────────────────────────────────
if model is None:
    st.error(
        "⚠️  **Model file not found.**  "
        "Place `xgboost_uhpfrc_model.pkl` in the same folder as `app.py` "
        "and restart the app."
    )
    st.stop()
else:
    st.markdown(
        '<div class="info-card">✅  Model loaded — '
        '<code>xgboost_uhpfrc_model.pkl</code>  |  Ready to predict.</div>',
        unsafe_allow_html=True
    )

# ── Input review table ────────────────────────────────────────
with st.expander("📋  Review current input values", expanded=False):
    import pandas as pd
    review = pd.DataFrame({
        "Feature"    : ["C","SF","QP","FA","SL","MK","S","QS","W","SP",
                         "Age","L","D","BV","PPV","GV","SSV"],
        "Description": ["Cement","Silica Fume","Quartz Powder","Fly Ash","Slag",
                         "Metakaolin","Sand","Quartz Sand","Water","Superplasticizer",
                         "Age (fixed)","Fibre Length","Fibre Diameter",
                         "Brass Fibre Vol.","PP Fibre Vol.","Glass Fibre Vol.",
                         "Steel Fibre Vol."],
        "Value"      : [C, SF, QP, FA, SL, MK, S, QS, W, SP,
                         28, L, D, BV, PPV, GV, SSV],
        "Unit"       : ["kg/m³"]*10 + ["days","mm","mm","%","%","%","%"]
    })
    st.dataframe(review, use_container_width=True, hide_index=True)

# ── Predict button ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("⚡  PREDICT COMPRESSIVE STRENGTH")

if predict_clicked:
    # Assemble feature vector — Age hardcoded at 28 days
    # Order must match training: C SF QP FA SL MK S QS W SP Age L D BV PPV GV SSV
    feature_vector = np.array([[
        C, SF, QP, FA, SL, MK,
        S, QS, W, SP,
        28,                  # Age – fixed at 28 days
        L, D, BV, PPV, GV, SSV
    ]])

    # Run inference
    prediction = float(model.predict(feature_vector)[0])
    label, color = classify_strength(prediction)

    # ── Result card ───────────────────────────────────────────
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted Compressive Strength</div>
            <div class="result-value">{prediction:.1f}</div>
            <div class="result-unit">MPa  ·  28-day cube / cylinder strength</div>
            <span class="strength-tag"
                  style="background:rgba(0,0,0,0.3);
                         border:1px solid {color};
                         color:{color};">
                {label}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Contextual guidance ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted CS", f"{prediction:.1f} MPa")
    with c2:
        wc_ratio = round(W / (C + SF + QP + FA + SL + MK + 1e-9), 3)
        st.metric("W/B Ratio", f"{wc_ratio:.3f}")
    with c3:
        total_fibre = BV + PPV + GV + SSV
        st.metric("Total Fibre Vol.", f"{total_fibre:.2f} %")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
    'color:#7D8590;text-align:center;">'
    'UHPFRC CS Predictor  ·  XGBoost  ·  NIT Tiruchirappalli  ·  '
    'For research use only — validate predictions against experimental data.'
    '</p>',
    unsafe_allow_html=True
)
