# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) module.
Handles data distributions and correlation matrices.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import friendly_name, render_footer


def render(raw_df):
    st.markdown("# Explore the Data")
    
    st.markdown(
        '<div class="info-box">'
        "Use this section to visually explore the dataset prior to model training. "
        "Reviewing feature distributions and correlation matrices can help identify outliers "
        "and reveal which variables have the strongest relationships with the target."
        "</div>",
        unsafe_allow_html=True,
    )

    # Prepare a numerical version of the dataset for the correlation matrix
    df_numeric = raw_df.copy()
    df_numeric["hour"] = df_numeric.hour.str.slice(1, 3).astype(int)
    df_numeric = df_numeric.drop(
        columns=["user", "screen_list", "enrolled_date", "first_open", "enrolled"],
        errors="ignore",
    )

    tab1, tab2, tab3 = st.tabs(
        ["Feature Distributions", "Correlation with Enrollment", "Full Correlation Matrix"]
    )

    # Tab 1: Histograms of numerical features
    with tab1:
        st.markdown("### Feature Distributions")
        st.caption(
            "These charts display the distribution of values for each numerical feature. "
            "They are useful for spotting skewed data or unexpected spikes."
        )
        
        cols = df_numeric.columns.tolist()
        n_cols = 3
        n_rows = (len(cols) + n_cols - 1) // n_cols

        fig_hist = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[friendly_name(c) for c in cols])
        
        for idx, col_name in enumerate(cols):
            r, c = idx // n_cols + 1, idx % n_cols + 1
            fig_hist.add_trace(
                go.Histogram(x=df_numeric[col_name], marker_color="#6C63FF", opacity=0.85, showlegend=False),
                row=r, col=c,
            )
            
        fig_hist.update_layout(
            height=280 * n_rows, 
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Tab 2: Feature correlation against the target variable
    with tab2:
        st.markdown("### Correlation with Enrollment")
        st.caption(
            "Features pointing to the right indicate a positive correlation with enrollment. "
            "Features pointing to the left indicate a negative correlation."
        )
        
        raw_enrolled = raw_df.copy()
        raw_enrolled["hour"] = raw_enrolled.hour.str.slice(1, 3).astype(int)
        corr_with = df_numeric.corrwith(raw_enrolled["enrolled"]).sort_values()
        corr_with.index = [friendly_name(i) for i in corr_with.index]

        fig_bar = go.Figure(
            go.Bar(
                x=corr_with.values, 
                y=corr_with.index, 
                orientation="h",
                marker=dict(
                    color=corr_with.values,
                    colorscale=[[0, "#FF6584"], [0.5, "#3a3f5c"], [1, "#6C63FF"]],
                    showscale=True, 
                    colorbar=dict(title="r"),
                ),
                hovertemplate="<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>",
            )
        )
        
        fig_bar.update_layout(
            height=500, 
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Pearson Correlation Coefficient",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tab 3: Heatmap of all pairwise correlations
    with tab3:
        st.markdown("### Full Correlation Matrix")
        st.caption(
            "This heatmap illustrates the pairwise correlations between all features. "
            "High correlation between two features may indicate redundancy."
        )
        
        corr_matrix = df_numeric.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_masked = corr_matrix.where(~mask)
        labels = [friendly_name(c) for c in corr_matrix.columns]

        fig_hm = go.Figure(
            go.Heatmap(
                z=corr_masked.values, 
                x=labels, 
                y=labels,
                colorscale="RdBu_r", 
                zmin=-1, 
                zmax=1,
                text=np.round(corr_masked.values, 2), 
                texttemplate="%{text}",
                textfont={"size": 9},
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation (r) = %{z:.3f}<extra></extra>",
            )
        )
        
        fig_hm.update_layout(
            height=700, 
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    render_footer()
