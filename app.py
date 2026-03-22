import numpy as np
import pandas as pd
import pickle
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='IPL Score Predictor',
    page_icon='🏏',
    layout='centered'
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e8eaf0;
}

/* Main background */
.stApp {
    background: radial-gradient(ellipse at top left, #0d1829 0%, #080c14 60%);
    min-height: 100vh;
}

/* Title */
.title-block {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.title-block h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.06em;
    background: linear-gradient(135deg, #f7b731 0%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1;
}
.title-block p {
    color: #7a8299;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    letter-spacing: 0.03em;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #f7b731;
    margin-bottom: 1rem;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #f7b731 0%, #ff6b35 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 8px 40px rgba(247, 183, 49, 0.25);
}
.result-box .label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(0,0,0,0.55);
    margin-bottom: 0.3rem;
}
.result-box .score {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: #080c14;
    line-height: 1;
    letter-spacing: 0.04em;
}
.result-box .sub {
    font-size: 0.85rem;
    color: rgba(0,0,0,0.5);
    margin-top: 0.4rem;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.08), transparent);
    margin: 1.5rem 0;
}

/* Selectbox & number input overrides */
div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"] > div {
    background-color: #111827 !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #f7b731 0%, #ff6b35 100%);
    color: #080c14;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
    margin-top: 0.5rem;
}
.stButton > button:hover { opacity: 0.88; }

/* Slider */
.stSlider > div > div { background: #1e2a3a !important; }

/* Error */
.error-box {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #f87171;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #3a4155;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-bottom: 2rem;
}
.footer a { color: #f7b731; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ── Load model & encoder ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = pickle.load(open('ipl.pkl',         'rb'))
    encoder = pickle.load(open('ipl_encoder.pkl', 'rb'))
    return model, encoder

try:
    model, encoder = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Constants ────────────────────────────────────────────────────────────────
TEAMS = [
    'Chennai Super Kings',
    'Delhi Daredevils',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

VENUES = [
    'Eden Gardens',
    'Wankhede Stadium',
    'M Chinnaswamy Stadium',
    'Feroz Shah Kotla',
    'MA Chidambaram Stadium',
    'Punjab Cricket Association Stadium, Mohali',
    'Rajiv Gandhi International Stadium, Uppal',
    'Sawai Mansingh Stadium',
    'DY Patil Stadium',
    'Subrata Roy Sahara Stadium'
]

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🏏 IPL Score Predictor</h1>
    <p>Live match state → Predicted final score · Powered by Random Forest (R² = 0.91)</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.markdown("""
    <div class="error-box">
        ⚠️ Model files not found (<code>ipl.pkl</code> / <code>ipl_encoder.pkl</code>).
        Please run the training notebook first to generate them.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Form ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">⚔️ Match Setup</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team', TEAMS, index=4)
with col2:
    bowling_team = st.selectbox('Bowling Team', TEAMS, index=0)

venue = st.selectbox('Venue / Stadium', VENUES)

if batting_team == bowling_team:
    st.markdown('<div class="error-box">⚠️ Batting and Bowling teams must be different.</div>',
                unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Match State ───────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📊 Current Match State</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    overs = st.number_input('Current Over', min_value=5.1, max_value=19.5,
                             value=10.0, step=0.1,
                             help='Minimum 5 overs required. Format: 10.3 = 10th over, 3rd ball')
with col4:
    runs = st.number_input('Runs Scored So Far', min_value=0, max_value=300,
                            value=68, step=1)

wickets = st.slider('Wickets Fallen', min_value=0, max_value=9, value=3)

col5, col6 = st.columns(2)
with col5:
    runs_last_5 = st.number_input('Runs in Last 5 Overs', min_value=0,
                                   max_value=120, value=29, step=1)
with col6:
    wickets_last_5 = st.number_input('Wickets in Last 5 Overs', min_value=0,
                                      max_value=5, value=1, step=1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button('🏏 Predict Final Score'):
    if batting_team == bowling_team:
        st.markdown('<div class="error-box">Please select different teams.</div>',
                    unsafe_allow_html=True)
    elif overs < 5.0:
        st.markdown('<div class="error-box">At least 5 overs must be completed.</div>',
                    unsafe_allow_html=True)
    else:
        # Feature engineering (must match training notebook)
        run_rate            = runs / overs
        req_run_rate_proxy  = runs_last_5 / 5
        wickets_remaining   = 10 - wickets
        balls_bowled        = int(overs) * 6 + round((overs % 1) * 10)
        balls_remaining     = 120 - balls_bowled

        input_df = pd.DataFrame([{
            'bat_team':             batting_team,
            'bowl_team':            bowling_team,
            'venue':                venue,
            'runs':                 runs,
            'wickets':              wickets,
            'overs':                overs,
            'runs_last_5':          runs_last_5,
            'wickets_last_5':       wickets_last_5,
            'run_rate':             run_rate,
            'req_run_rate_proxy':   req_run_rate_proxy,
            'wickets_remaining':    wickets_remaining,
            'balls_bowled':         balls_bowled,
            'balls_remaining':      balls_remaining
        }])

        try:
            encoded   = encoder.transform(input_df)
            pred      = int(round(model.predict(encoded)[0]))
            score_low = pred - 5
            score_hi  = pred + 5

            st.markdown(f"""
            <div class="result-box">
                <div class="label">Predicted Final Score</div>
                <div class="score">{score_low} – {score_hi}</div>
                <div class="sub">
                    {batting_team} batting at {venue}<br>
                    Over {overs} · {runs}/{wickets} · Run Rate {run_rate:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="error-box">Prediction error: {e}</div>',
                        unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    Built by <strong>Suyog Satibawane</strong> ·
    <a href="https://github.com/suyog0229/Suyog-Satibawane-Projects" target="_blank">GitHub</a> ·
    Random Forest · R² = 0.91 · 50,000+ IPL records
</div>
""", unsafe_allow_html=True)
