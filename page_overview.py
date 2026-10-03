# -*- coding: utf-8 -*-
"""
Overview module. Introduces the project and shows high-level metrics.
"""

import streamlit as st
import plotly.graph_objects as go
from utils import render_footer


def render(raw_df, top_screens):
    st.markdown("# Directing Customers to Subscription")
    
    # Project introduction text
    st.markdown(
        '<div class="info-box">'
        "<strong>Project Overview</strong><br>"
        "This dashboard analyzes how users navigate through a mobile app and predicts "
        "whether they will <strong>subscribe (enroll)</strong> to the premium service. "
        "It uses machine learning to identify which in-app behaviors serve as the strongest "
        "signals for conversion.<br><br>"
        "Use the sidebar to explore the data, inspect the feature engineering pipeline, "
        "train the model, or make predictions on new user profiles."
        "</div>",
        unsafe_allow_html=True,
    )

    # Key Performance Indicators
    enrolled_rate = raw_df["enrolled"].mean() * 100
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Users", f"{len(raw_df):,}", help="Total number of unique users in the dataset")
    col2.metric("Raw Features", raw_df.shape[1], help="Number of columns in the original CSV file")
    col3.metric("Enrollment Rate", f"{enrolled_rate:.1f}%", help="Percentage of users who completed a subscription")
    col4.metric("Top Screens Tracked", len(top_screens), help="Number of specific app screens tracked individually")

    st.markdown("")
    left_col, right_col = st.columns([3, 2])

    # Raw Data table
    with left_col:
        st.markdown("### Raw Data Preview")
        st.caption("First 20 rows of the original dataset. Scroll right to view all columns.")
        st.dataframe(raw_df.head(20), use_container_width=True, height=380)

    # Enrollment donut chart
    with right_col:
        st.markdown("### Enrollment Split")
        st.caption("Distribution of users who subscribed versus those who did not.")
        
        enrolled_counts = raw_df["enrolled"].value_counts()
        
        fig_donut = go.Figure(
            go.Pie(
                labels=["Did Not Subscribe", "Subscribed"],
                values=[enrolled_counts.get(0, 0), enrolled_counts.get(1, 0)],
                hole=0.6,
                marker=dict(colors=["#3a3f5c", "#6C63FF"]),
                textinfo="percent+label",
                textfont_size=13,
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        )
        
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            showlegend=False,
            margin=dict(t=10, b=10),
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)

    # Dictionary explaining the meaning of each column
    with st.expander("Column Glossary - Variable Definitions"):
        st.markdown(
            """
| Column | Description |
|--------|-------------|
| `user` | Unique identifier for each user |
| `first_open` | Date and time the user initially opened the application |
| `dayofweek` | Day of the week corresponding to the first open (0 = Monday, 6 = Sunday) |
| `hour` | Hour of the day the application was first opened |
| `age` | Age of the user at the time of registration |
| `screen_list` | Comma-separated list representing every screen the user navigated to |
| `numscreens` | Total count of screens visited during the session |
| `minigame` | Binary indicator for whether the user played the included mini-game (0 or 1) |
| `used_premium_feature` | Binary indicator for whether the user tested a premium feature (0 or 1) |
| `enrolled` | **Target variable**: binary indicator for user subscription status (0 or 1) |
| `enrolled_date` | Date and time of subscription (empty if the user did not enroll) |
| `liked` | Binary indicator for whether the user liked any application content (0 or 1) |
"""
        )

    render_footer()
