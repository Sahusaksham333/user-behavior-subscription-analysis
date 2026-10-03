# -*- coding: utf-8 -*-
"""
Main application entry point. Handles routing and basic configuration.
"""

import streamlit as st
import warnings
from utils import load_raw_data, load_top_screens, feature_engineer
import page_overview
import page_explore
import page_features
import page_model
import page_predict

warnings.filterwarnings("ignore")

# Basic page configuration
st.set_page_config(
    page_title="App Subscription Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling the dashboard
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* metric cards styling */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e2030 0%, #252940 100%);
    border: 1px solid rgba(108, 99, 255, .25);
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, .2);
}
div[data-testid="stMetric"] label { color: #a0a4b8 !important; font-weight: 500; font-size: 0.85rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #6C63FF !important; font-weight: 700; }

/* sidebar background */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #12141e 0%, #1a1d29 100%); }

/* clean up typography */
h1 { letter-spacing: -0.03em; }
h2, h3 { letter-spacing: -0.02em; }

/* tabs font weight */
button[data-baseweb="tab"] { font-weight: 600 !important; font-size: 0.95rem !important; }

/* informational boxes */
.info-box {
    background: linear-gradient(135deg, #1a1f35 0%, #1e2340 100%);
    border-left: 4px solid #6C63FF;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 12px 0 20px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #c8cad8;
}
.info-box strong { color: #a8a2ff; }

/* tags/badges for steps */
.step-badge {
    display: inline-block;
    background: #6C63FF;
    color: white;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 8px;
    vertical-align: middle;
}

/* grouped feature cards */
.feature-group {
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.feature-group h4 { margin: 0 0 8px 0; color: #a8a2ff; font-size: 0.95rem; }

/* footer layout */
.footer-text {
    text-align: center;
    color: #555;
    font-size: 0.78rem;
    padding: 30px 0 10px 0;
    border-top: 1px solid rgba(108, 99, 255, .1);
    margin-top: 40px;
}
</style>
""",
    unsafe_allow_html=True,
)


# Sidebar navigation
with st.sidebar:
    st.markdown("# Dashboard Navigation")
    st.caption("Select a section to explore.")
    st.markdown("")
    
    page = st.radio(
        "Section",
        ["Overview", "Explore Data", "Feature Engineering", "Train Model", "Predict User"],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    st.markdown(
        "<p style='color:#6b6f80; font-size:0.8rem; text-align:center;'>"
        "Developed by <strong style='color:#a8a2ff;'>Saksham Sahu</strong></p>",
        unsafe_allow_html=True,
    )


# Load data and run the preprocessing pipeline once
raw_df = load_raw_data()
top_screens = load_top_screens()
clean_df = feature_engineer(raw_df, top_screens)


# Route the user to the correct page module
if page == "Overview":
    page_overview.render(raw_df, top_screens)
elif page == "Explore Data":
    page_explore.render(raw_df)
elif page == "Feature Engineering":
    page_features.render(raw_df, clean_df)
elif page == "Train Model":
    page_model.render(clean_df)
elif page == "Predict User":
    page_predict.render(clean_df)
