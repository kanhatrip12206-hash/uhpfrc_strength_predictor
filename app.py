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
# CUSTOM CSS  –  blueprint / engineering light theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Fraunces:opsz,wght@9..144,300;9..144,700;9..144,900&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Palette ── */
:root {
    --bg           : #F4F1EB;
    --bg-card      : #FFFFFF;
    --bg-sidebar   : #1B2A3B;
    --ink          : #1B2A3B;
    --ink-muted    : #5A6A7A;
    --ink-faint    : #A0AEBB;
    --gold         : #C8973A;
    --gold-light   : #F5E6C8;
    --gold-glow    : rgba(200, 151, 58, 0.18);
    --green        : #1E7D57;
    --border       : #DDD8CE;
    --shadow       : 0 2px 12px rgba(27,42,59,0.08);
    --shadow-lg    : 0 8px 32px rgba(27,42,59,0.14);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family : 'IBM Plex Sans', sans-serif !important;
    background  : var(--bg) !important;
    color       : var(--ink) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background : var(--bg-sidebar) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color         : #8A9BAD !important;
    font-family   : 'IBM Plex Mono', monospace !important;
    font-size     : 0.68rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* Sidebar number inputs */
[data-testid="stSidebar"] input[type="number"] {
    background    : rgba(255,255,255,0.05) !important;
    border        : 1px solid rgba(255,255,255,0.1) !important;
    border-radius : 4px !important;
    color         : #DCE4EC !important;
    font-family   : 'IBM Plex Mono', monospace !important;
    font-size     : 0.84rem !important;
}
[data-testid="stSidebar"] input[type="number"]:focus {
    border-color  : var(--gold) !important;
    box-shadow    : 0 0 0 2px rgba(200,151,58,0.2) !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #7A8FA0 !important; font-size: 0.77rem !important; }

/* ── Sidebar custom blocks ── */
.sb-header {
    background    : var(--gold);
    padding       : 1.4rem 1.4rem 1.2rem;
    margin-bottom : 0.2rem;
}
.sb-header-title {
    font-family   : 'Fraunces', serif;
    font-size     : 1.1rem;
    font-weight   : 700;
    color         : #1B2A3B;
    line-height   : 1.25;
}
.sb-header-sub {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.63rem;
    color         : rgba(27,42,59,0.6);
    margin-top    : 0.25rem;
    letter-spacing: 0.05em;
}
.sb-section {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color         : var(--gold);
    background    : rgba(200,151,58,0.1);
    border-left   : 2px solid var(--gold);
    padding       : 0.38rem 0.8rem;
    margin        : 1.1rem 0 0.45rem 0;
    border-radius : 0 3px 3px 0;
}
.sb-age-badge {
    display       : flex;
    align-items   : center;
    gap           : 0.6rem;
    background    : rgba(255,255,255,0.05);
    border        : 1px solid rgba(255,255,255,0.1);
    border-radius : 6px;
    padding       : 0.65rem 0.9rem;
    margin-top    : 0.9rem;
}
.sb-age-num {
    font-family   : 'Fraunces', serif;
    font-size     : 1.9rem;
    font-weight   : 900;
    color         : var(--gold);
    line-height   : 1;
}
.sb-age-text {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.62rem;
    color         : #7A8FA0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    line-height   : 1.5;
}

/* ── Main area ── */
.main-wrapper { padding: 2.4rem 2.8rem 3rem; max-width: 1100px; }

/* ── Page header ── */
.page-eyebrow {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.65rem;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color         : var(--gold);
    margin-bottom : 0.45rem;
}
.page-title {
    font-family   : 'Fraunces', serif;
    font-size     : 3rem;
    font-weight   : 900;
    line-height   : 1.02;
    color         : var(--ink);
    letter-spacing: -0.03em;
}
.page-title span { color: var(--gold); }
.page-desc {
    font-size     : 0.87rem;
    color         : var(--ink-muted);
    line-height   : 1.65;
    max-width     : 470px;
    margin-top    : 0.55rem;
}
.stat-row {
    display       : flex;
    gap           : 2rem;
    margin-top    : 1.1rem;
    flex-wrap     : wrap;
}
.stat-item { display: flex; flex-direction: column; }
.stat-num {
    font-family   : 'Fraunces', serif;
    font-size     : 1.55rem;
    font-weight   : 700;
    color         : var(--ink);
    line-height   : 1;
}
.stat-lbl {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.58rem;
    letter-spacing: 0.1em;
    color         : var(--ink-faint);
    text-transform: uppercase;
    margin-top    : 0.12rem;
}
.rule { border: none; border-top: 1px solid var(--border); margin: 1.6rem 0; }

/* ── Status banner ── */
.status-ok {
    display       : flex;
    align-items   : center;
    gap           : 0.7rem;
    background    : #EAF4EE;
    border        : 1px solid #B8DEC7;
    border-radius : 6px;
    padding       : 0.6rem 1rem;
    font-size     : 0.78rem;
    color         : var(--green);
    font-family   : 'IBM Plex Mono', monospace;
    margin-bottom : 1.4rem;
}
.status-dot {
    width         : 7px;
    height        : 7px;
    border-radius : 50%;
    background    : var(--green);
    flex-shrink   : 0;
    animation     : pulse 2.2s infinite;
}
@keyframes pulse { 0%, 100% { opacity:1; } 50% { opacity:0.35; } }

/* ── Card ── */
.card {
    background    : var(--bg-card);
    border        : 1px solid var(--border);
    border-radius : 10px;
    padding       : 1.4rem 1.5rem;
    box-shadow    : var(--shadow);
    margin-bottom : 1rem;
}
.card-title {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.62rem;
    letter-spacing: 0.17em;
    text-transform: uppercase;
    color         : var(--ink-faint);
    margin-bottom : 0.9rem;
    padding-bottom: 0.55rem;
    border-bottom : 1px solid var(--border);
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background    : var(--ink) !important;
    color         : #FFFFFF !important;
    font-family   : 'IBM Plex Sans', sans-serif !important;
    font-weight   : 600 !important;
    font-size     : 0.9rem !important;
    letter-spacing: 0.04em !important;
    border        : 2px solid var(--ink) !important;
    border-radius : 6px !important;
    padding       : 0.68rem 1.8rem !important;
    width         : 100% !important;
    transition    : all 0.18s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background    : var(--gold) !important;
    border-color  : var(--gold) !important;
    color         : var(--ink) !important;
    transform     : translateY(-1px) !important;
    box-shadow    : 0 4px 14px var(--gold-glow) !important;
}

/* ── Result panel (dark card, right column) ── */
.result-panel {
    background    : var(--ink);
    border-radius : 10px;
    padding       : 2.2rem 1.8rem 1.8rem;
    box-shadow    : var(--shadow-lg);
    text-align    : center;
    position      : relative;
    overflow      : hidden;
    animation     : slideUp 0.35s ease;
}
.result-panel::before {
    content       : '';
    position      : absolute;
    top:0; left:0; right:0;
    height        : 3px;
    background    : linear-gradient(90deg, var(--gold), #E8B86D, var(--gold));
}
.result-eyebrow {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.6rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color         : var(--gold);
    margin-bottom : 0.9rem;
}
.result-number {
    font-family   : 'Fraunces', serif;
    font-size     : 5.8rem;
    font-weight   : 900;
    color         : #FFFFFF;
    line-height   : 1;
    letter-spacing: -0.04em;
}
.result-mpa {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.95rem;
    color         : rgba(255,255,255,0.38);
    margin-top    : 0.15rem;
    letter-spacing: 0.12em;
}
.result-badge {
    display       : inline-block;
    margin-top    : 1.1rem;
    padding       : 0.32rem 1.1rem;
    border-radius : 4px;
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.68rem;
    letter-spacing: 0.07em;
    font-weight   : 500;
}
.result-divider {
    border        : none;
    border-top    : 1px solid rgba(255,255,255,0.07);
    margin        : 1.5rem 0 1.25rem;
}
.mini-row { display:flex; justify-content:space-around; }
.mini-metric { text-align:center; }
.mini-metric-val {
    font-family   : 'Fraunces', serif;
    font-size     : 1.45rem;
    font-weight   : 700;
    color         : #FFFFFF;
    line-height   : 1;
}
.mini-metric-lbl {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.56rem;
    color         : rgba(255,255,255,0.3);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top    : 0.22rem;
}

/* ── Waiting state ── */
.waiting-panel {
    background    : var(--bg-card);
    border        : 2px dashed var(--border);
    border-radius : 10px;
    padding       : 3.5rem 2rem;
    text-align    : center;
}
.waiting-icon { font-size:2.8rem; opacity:0.2; margin-bottom:0.8rem; }
.waiting-text {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.7rem;
    letter-spacing: 0.12em;
    color         : var(--ink-faint);
    text-transform: uppercase;
}
.waiting-hint {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.63rem;
    color         : #C0CBDA;
    margin-top    : 0.5rem;
    letter-spacing: 0.05em;
}

/* ── Footer ── */
.app-footer {
    font-family   : 'IBM Plex Mono', monospace;
    font-size     : 0.63rem;
    color         : var(--ink-faint);
    text-align    : center;
    letter-spacing: 0.06em;
    padding       : 2rem 0 1rem;
    border-top    : 1px solid var(--border);
    margin-top    : 2.5rem;
}

@keyframes slideUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
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
    """Return a (label, hex_color, bg_color) tuple based on CS value (MPa)."""
    if cs >= 150:
        return "Ultra-High Strength  ≥ 150 MPa", "#C8973A", "rgba(200,151,58,0.15)"
    elif cs >= 120:
        return "Very High Strength   120–150 MPa", "#4FC3F7", "rgba(79,195,247,0.15)"
    elif cs >= 100:
        return "High Strength        100–120 MPa", "#FFB74D", "rgba(255,183,77,0.15)"
    else:
        return "Moderate Strength    < 100 MPa",   "#FF6B6B", "rgba(255,107,107,0.15)"


# ─────────────────────────────────────────────────────────────
# SIDEBAR  –  input controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
        <div class="sb-header">
            <div class="sb-header-title">Mix Design<br>Parameters</div>
            <div class="sb-header-sub">All quantities in kg/m³ unless noted</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Binder Materials</div>', unsafe_allow_html=True)
    C  = st.number_input("C  – Cement",               min_value=0.0, value=800.0,  step=10.0, format="%.2f")
    SF = st.number_input("SF – Silica Fume",           min_value=0.0, value=150.0,  step=5.0,  format="%.2f")
    QP = st.number_input("QP – Quartz Powder",         min_value=0.0, value=0.0,    step=5.0,  format="%.2f")
    FA = st.number_input("FA – Fly Ash",               min_value=0.0, value=0.0,    step=5.0,  format="%.2f")
    SL = st.number_input("SL – Slag",                  min_value=0.0, value=0.0,    step=5.0,  format="%.2f")
    MK = st.number_input("MK – Metakaolin",            min_value=0.0, value=0.0,    step=5.0,  format="%.2f")

    st.markdown('<div class="sb-section">Aggregates</div>', unsafe_allow_html=True)
    S  = st.number_input("S  – Sand",                  min_value=0.0, value=1100.0, step=10.0, format="%.2f")
    QS = st.number_input("QS – Quartz Sand",           min_value=0.0, value=0.0,    step=10.0, format="%.2f")

    st.markdown('<div class="sb-section">Water & Admixtures</div>', unsafe_allow_html=True)
    W  = st.number_input("W  – Water",                 min_value=0.0, value=180.0,  step=5.0,  format="%.2f")
    SP = st.number_input("SP – Superplasticizer",      min_value=0.0, value=20.0,   step=1.0,  format="%.2f")

    st.markdown('<div class="sb-section">Fibre Parameters</div>', unsafe_allow_html=True)
    L   = st.number_input("L   – Fibre Length (mm)",   min_value=0.0, value=13.0,   step=0.5,  format="%.2f")
    D   = st.number_input("D   – Fibre Diameter (mm)", min_value=0.0, value=0.2,    step=0.01, format="%.3f")
    BV  = st.number_input("BV  – Brass Fibre Vol. (%)",min_value=0.0, value=0.0,    step=0.1,  format="%.2f")
    PPV = st.number_input("PPV – PP Fibre Vol. (%)",   min_value=0.0, value=0.0,    step=0.1,  format="%.2f")
    GV  = st.number_input("GV  – Glass Fibre Vol. (%)",min_value=0.0, value=0.0,    step=0.1,  format="%.2f")
    SSV = st.number_input("SSV – Steel Fibre Vol. (%)",min_value=0.0, value=2.0,    step=0.1,  format="%.2f")

    st.markdown("""
        <div class="sb-age-badge">
            <div class="sb-age-num">28</div>
            <div class="sb-age-text">Days curing<br>Fixed · IS 516 / ASTM C39</div>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# ── Page header ──────────────────────────────────────────────
st.markdown('<div class="page-eyebrow">XGBoost · Machine Learning · UHPFRC</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="page-title">Compressive<br><span>Strength</span> Predictor</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="page-desc">'
    'Predicts 28-day compressive strength of Ultra-High Performance '
    'Fibre-Reinforced Concrete from 17 mix design parameters using a '
    'gradient-boosted regression model trained on 864 experimental records.'
    '</div>',
    unsafe_allow_html=True
)
st.markdown("""
    <div class="stat-row">
        <div class="stat-item">
            <span class="stat-num">864</span>
            <span class="stat-lbl">Training samples</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">17</span>
            <span class="stat-lbl">Input features</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">28d</span>
            <span class="stat-lbl">Curing age</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">XGB</span>
            <span class="stat-lbl">Algorithm</span>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ── Model status ─────────────────────────────────────────────
if model is None:
    st.error(
        "⚠️  **Model file not found.**  "
        "Place `xgboost_uhpfrc_model.pkl` in the same folder as `app.py` "
        "and restart the app."
    )
    st.stop()
else:
    st.markdown(
        '<div class="status-ok">'
        '<div class="status-dot"></div>'
        'Model loaded &nbsp;—&nbsp; <code>xgboost_uhpfrc_model.pkl</code>'
        '&nbsp;·&nbsp; Ready to predict'
        '</div>',
        unsafe_allow_html=True
    )

# ── Two-column layout: table + result ────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown(
        '<div class="card"><div class="card-title">📋 &nbsp;Current Input Summary</div>',
        unsafe_allow_html=True
    )
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
    st.dataframe(review, use_container_width=True, hide_index=True, height=310)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡  PREDICT COMPRESSIVE STRENGTH")

with col_right:
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
        label, color, bg_color = classify_strength(prediction)

        # Derived display metrics
        wc_ratio    = round(W / (C + SF + QP + FA + SL + MK + 1e-9), 3)
        total_fibre = round(BV + PPV + GV + SSV, 2)

        # Result card
        st.markdown(f"""
            <div class="result-panel">
                <div class="result-eyebrow">Predicted Compressive Strength</div>
                <div class="result-number">{prediction:.1f}</div>
                <div class="result-mpa">MPa &nbsp;·&nbsp; 28-day strength</div>
                <span class="result-badge"
                      style="background:{bg_color};
                             color:{color};
                             border:1px solid {color};">
                    {label}
                </span>
                <hr class="result-divider">
                <div class="mini-row">
                    <div class="mini-metric">
                        <div class="mini-metric-val">{prediction:.1f}</div>
                        <div class="mini-metric-lbl">CS (MPa)</div>
                    </div>
                    <div class="mini-metric">
                        <div class="mini-metric-val">{wc_ratio:.3f}</div>
                        <div class="mini-metric-lbl">W/B Ratio</div>
                    </div>
                    <div class="mini-metric">
                        <div class="mini-metric-val">{total_fibre:.2f}%</div>
                        <div class="mini-metric-lbl">Total Fibre</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:
        # Waiting state shown before first prediction
        st.markdown("""
            <div class="waiting-panel">
                <div class="waiting-icon">🏗️</div>
                <div class="waiting-text">Awaiting Prediction</div>
                <div class="waiting-hint">
                    Set mix parameters in the sidebar<br>then click Predict
                </div>
            </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">'
    'UHPFRC CS Predictor &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; '
    'NIT Tiruchirappalli &nbsp;·&nbsp; '
    'For research use only — validate predictions against experimental data.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
