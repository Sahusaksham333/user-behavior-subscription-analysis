# -*- coding: utf-8 -*-
"""
Predict User module.
Provides an interactive interface to make real-time predictions using the trained model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utils import friendly_name, FEATURE_HELP, render_footer

def render(clean_df):
    st.markdown("# Enrollment Prediction Tool")
    
    st.markdown(
        '<div class="info-box">'
        "Adjust the parameters below to simulate different user profiles. "
        "The model will generate a real-time prediction detailing the probability that the hypothetical user will subscribe. "
        "Hover over the help icon next to a parameter for its definition."
        "</div>",
        unsafe_allow_html=True,
    )

    # Train the model on the full dataset before predicting
    model_df = clean_df.copy()
    response = model_df["enrolled"]
    model_df = model_df.drop(columns="enrolled")
    X_all = model_df.drop(columns=["user"])

    sc = StandardScaler()
    X_all_sc = sc.fit_transform(X_all)
    
    clf = LogisticRegression(random_state=0, penalty="l1", solver="liblinear", max_iter=1000)
    clf.fit(X_all_sc, response)

    feature_names = X_all.columns.tolist()

    # Organize features into logical groups for the user interface
    demographic_feats = [f for f in feature_names if f in ["dayofweek", "hour", "age"]]
    behaviour_feats = [f for f in feature_names if f in ["numscreens", "minigame", "used_premium_feature", "liked"]]
    funnel_feats = [f for f in feature_names if f in ["SavingCount", "CMCount", "CCCount", "LoansCount", "Other"]]
    screen_feats = [f for f in feature_names if f not in demographic_feats + behaviour_feats + funnel_feats]

    input_values = {}

    def render_slider_group(title, feats, cols_count=3):
        """Helper to render a visually distinct group of input sliders."""
        if not feats:
            return
            
        st.markdown(f"#### {title}")
        cols = st.columns(cols_count)
        
        for idx, feat in enumerate(feats):
            col = cols[idx % cols_count]
            
            # Identify bounds based on the full dataset
            mn, mx = float(X_all[feat].min()), float(X_all[feat].max())
            med = float(X_all[feat].median())
            
            label = friendly_name(feat)
            tip = FEATURE_HELP.get(feat, "")

            # Adjust the slider input type based on the feature data type
            if X_all[feat].nunique() <= 10:
                input_values[feat] = col.slider(label, int(mn), int(max(mx, mn + 1)), int(med), help=tip)
            elif X_all[feat].dtype in [np.int64, np.int32]:
                input_values[feat] = col.slider(label, int(mn), int(max(mx, mn + 1)), int(med), help=tip)
            else:
                input_values[feat] = col.slider(label, mn, mx, med, help=tip)

    # Render groups
    render_slider_group("Demographics", demographic_feats, 3)
    render_slider_group("In-App Behavior", behaviour_feats, 4)
    render_slider_group("Funnel Progress", funnel_feats, 5)

    if screen_feats:
        with st.expander(f"Individual Screen Flags ({len(screen_feats)} screens) - expand to configure"):
            st.caption("Binary indicators (0 or 1) representing whether the user navigated to these specific screens.")
            
            cols = st.columns(6)
            for idx, feat in enumerate(screen_feats):
                col = cols[idx % 6]
                mn, mx = int(X_all[feat].min()), int(max(X_all[feat].max(), 1))
                med = int(X_all[feat].median())
                input_values[feat] = col.slider(feat, mn, mx, med, key=f"screen_{feat}")

    st.markdown("---")
    
    # Process the form and execute prediction
    if st.button("Predict Subscription Probability", use_container_width=True, type="primary"):
        # Format input vector
        user_vec = np.array([[input_values[f] for f in feature_names]])
        user_vec_sc = sc.transform(user_vec)
        
        prob = clf.predict_proba(user_vec_sc)[0][1]
        pred = clf.predict(user_vec_sc)[0]

        st.markdown("")
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        
        with col_res1:
            st.metric("Calculated Probability", f"{prob:.1%}")
            
        with col_res2:
            # Render probability gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Enrollment Likelihood", "font": {"size": 18, "color": "#c8cad8"}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": "#6C63FF"},
                    "steps": [
                        {"range": [0, 30], "color": "#FF6584"},
                        {"range": [30, 70], "color": "#FFC857"},
                        {"range": [70, 100], "color": "#43E8D8"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8, "value": prob * 100,
                    },
                },
                number={"suffix": "%", "font": {"size": 36}},
            ))
            fig_gauge.update_layout(
                height=300, 
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_res3:
            # Provide an actionable interpretation
            if pred == 1:
                st.success("### Likely to Subscribe")
                st.caption("Based on the provided metrics, this user profile indicates a strong likelihood of subscribing.")
            else:
                st.error("### Unlikely to Subscribe")
                st.caption("Based on the provided metrics, this user profile does not demonstrate a strong likelihood of subscribing.")

    render_footer()
