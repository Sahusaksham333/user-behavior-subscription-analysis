# -*- coding: utf-8 -*-
"""
Feature Engineering module.
Explains data transformations and displays the final cleaned data set.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import render_footer


def render(raw_df, clean_df):
    st.markdown("# Feature Engineering")
    
    st.markdown(
        '<div class="info-box">'
        "Raw application data often requires significant transformation before it can be effectively "
        "processed by a machine learning model. This page outlines the data cleaning and feature extraction pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    # Explanation of the pipeline steps
    st.markdown(
        """
<span class="step-badge">Step 1</span> **Extract the hour**: Parse the time string to create a numeric hour column (0-23).

<span class="step-badge">Step 2</span> **Apply 48-hour cutoff**: Treat any user who enrolled more than 48 hours after opening the app as a non-enrollment. This isolates users influenced by their first impression.

<span class="step-badge">Step 3</span> **Screen one-hot encoding**: Iterate through the top 58 screens and create binary indicators for each.

<span class="step-badge">Step 4</span> **Funnel aggregation**: Group related screens together into specific funnel counts, such as Savings, Credit Management, Credit Card, and Loans.

<span class="step-badge">Step 5</span> **Cleanup**: Remove raw text columns and dates that the model cannot process directly.
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    
    # Justification for the 48-hour cutoff
    st.markdown("### The 48-Hour Cutoff Logic")
    st.caption(
        "The following histogram illustrates the time delta between the first app open and enrollment. "
        "The vast majority of subscriptions occur within the first 48 hours. Enrollments after this window "
        "tend to be sparse and introduce noise into the model."
    )
    
    temp = raw_df.copy()
    temp["first_open"] = pd.to_datetime(temp["first_open"])
    temp["enrolled_date"] = pd.to_datetime(temp["enrolled_date"], errors="coerce")
    temp["diff_h"] = (temp.enrolled_date - temp.first_open).dt.total_seconds() / 3600

    fig_diff = go.Figure()
    fig_diff.add_trace(
        go.Histogram(
            x=temp["diff_h"].dropna(), 
            nbinsx=100,
            marker_color="#6C63FF", 
            opacity=0.8, 
            name="Enrolled users",
            hovertemplate="Hours: %{x:.0f}<br>Count: %{y}<extra></extra>",
        )
    )
    
    fig_diff.add_vline(
        x=48, 
        line_dash="dash", 
        line_color="#FF6584",
        annotation_text="48 Hour Cutoff", 
        annotation_font_color="#FF6584"
    )
    
    fig_diff.update_layout(
        xaxis_title="Hours Between First Open and Enrollment",
        yaxis_title="Number of Users",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    st.plotly_chart(fig_diff, use_container_width=True)

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # Preview of the cleaned dataframe
    with col1:
        st.markdown("### Cleaned Data Preview")
        st.caption("This table represents the data as it is passed into the model.")
        st.dataframe(clean_df.head(20), use_container_width=True, height=400)
        
    # Summary statistics for the cleaned dataset
    with col2:
        st.markdown("### Statistical Summary")
        st.caption("Distribution metrics for the cleaned features.")
        st.dataframe(clean_df.describe().T.style.format("{:.2f}"), use_container_width=True, height=400)

    # Distributions for engineered funnel features
    st.markdown("### Funnel Feature Distributions")
    st.caption("Distributions detailing how many screens users typically visit within each designated category.")
    
    funnel_cols = ["SavingCount", "CMCount", "CCCount", "LoansCount"]
    funnel_labels = ["Savings", "Credit Management", "Credit Card", "Loans"]
    existing = [(f, l) for f, l in zip(funnel_cols, funnel_labels) if f in clean_df.columns]
    
    if existing:
        fig_f = make_subplots(rows=1, cols=len(existing), subplot_titles=[l for _, l in existing])
        palette = ["#6C63FF", "#FF6584", "#43E8D8", "#FFC857"]
        
        for i, (fn, _) in enumerate(existing):
            fig_f.add_trace(
                go.Histogram(x=clean_df[fn], marker_color=palette[i], opacity=0.85, showlegend=False),
                row=1, col=i + 1,
            )
            
        fig_f.update_layout(
            height=320, 
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_f, use_container_width=True)

    render_footer()
