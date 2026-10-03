# -*- coding: utf-8 -*-
"""
Shared utilities for data loading, feature engineering, constants, and helpers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os


# Data loading functions, cached to prevent reloading on every interaction
@st.cache_data(show_spinner="Loading dataset, please wait...")
def load_raw_data():
    data_path = os.path.join(os.path.dirname(__file__), "appdata10.csv")
    return pd.read_csv(data_path)


@st.cache_data(show_spinner="Loading screen definitions...")
def load_top_screens():
    ts_path = os.path.join(os.path.dirname(__file__), "top_screens.csv")
    return pd.read_csv(ts_path).top_screens.values


@st.cache_data(show_spinner="Running feature engineering pipeline...")
def feature_engineer(df_raw, top_screens):
    """
    Mirrors the original EDA script: handles date parsing, extracts screens, 
    and groups them into defined funnels.
    """
    dataset = df_raw.copy()

    # Extract the hour from the time string
    dataset["hour"] = dataset.hour.str.slice(1, 3).astype(int)
    dataset["first_open"] = pd.to_datetime(dataset["first_open"])
    dataset["enrolled_date"] = pd.to_datetime(dataset["enrolled_date"], errors="coerce")

    # If the enrollment happened more than 48 hours after first open, 
    # we count it as a non-enrollment for the sake of this model
    dataset["difference"] = (
        (dataset.enrolled_date - dataset.first_open).dt.total_seconds() / 3600
    )
    dataset.loc[dataset.difference > 48, "enrolled"] = 0
    dataset = dataset.drop(columns=["enrolled_date", "difference", "first_open"])

    # Parse the comma-separated screen list into separate binary columns
    dataset["screen_list"] = dataset.screen_list.astype(str) + ","
    for sc in top_screens:
        dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
        dataset["screen_list"] = dataset.screen_list.str.replace(sc + ",", "", regex=False)
    
    # Count any leftover screens that weren't in the top list
    dataset["Other"] = dataset.screen_list.str.count(",")
    dataset = dataset.drop(columns=["screen_list"])

    # Aggregate specific screens into functional groups (funnels)
    funnel_groups = {
        "SavingCount": ["Saving1", "Saving2", "Saving2Amount", "Saving4", "Saving5", "Saving6", "Saving7", "Saving8", "Saving9", "Saving10"],
        "CMCount": ["Credit1", "Credit2", "Credit3", "Credit3Container", "Credit3Dashboard"],
        "CCCount": ["CC1", "CC1Category", "CC3"],
        "LoansCount": ["Loan", "Loan2", "Loan3", "Loan4"],
    }

    for label, screens in funnel_groups.items():
        present = [s for s in screens if s in dataset.columns]
        dataset[label] = dataset[present].sum(axis=1)
        dataset = dataset.drop(columns=present, errors="ignore")

    return dataset


# Mapping dictionaries to convert column names into human-readable labels
FEATURE_LABELS = {
    "dayofweek": "Day of Week (0=Mon to 6=Sun)",
    "hour": "Hour of Day (0-23)",
    "age": "User Age",
    "numscreens": "Total Screens Visited",
    "minigame": "Played Mini-Game? (0/1)",
    "used_premium_feature": "Used Premium Feature? (0/1)",
    "liked": "Liked Content? (0/1)",
    "SavingCount": "Savings Screens Visited",
    "CMCount": "Credit Mgmt Screens Visited",
    "CCCount": "Credit Card Screens Visited",
    "LoansCount": "Loan Screens Visited",
    "Other": "Other Screens Visited",
}

FEATURE_HELP = {
    "dayofweek": "Which day of the week the user first opened the app (Monday=0, Sunday=6).",
    "hour": "What hour (0-23) the user first opened the app.",
    "age": "The user's age at the time of signup.",
    "numscreens": "How many total screens the user navigated through in their session.",
    "minigame": "Whether the user played the in-app mini-game (1 = yes, 0 = no).",
    "used_premium_feature": "Whether the user tried a premium feature during the session.",
    "liked": "Whether the user 'liked' any content in the app.",
    "SavingCount": "Number of savings-related screens the user visited.",
    "CMCount": "Number of credit management screens visited.",
    "CCCount": "Number of credit card screens visited.",
    "LoansCount": "Number of loan screens visited.",
    "Other": "Count of screens not in the top-58 most popular screens.",
}


def friendly_name(feat):
    """Returns a more readable label for a given feature name."""
    return FEATURE_LABELS.get(feat, feat.replace("_", " ").title())


def render_footer():
    """Renders a simple footer at the bottom of the page."""
    st.markdown(
        '<div class="footer-text">Developed by <strong>Saksham Sahu</strong> | '
        'Powered by Streamlit & Scikit-learn</div>',
        unsafe_allow_html=True,
    )
