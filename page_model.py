# -*- coding: utf-8 -*-
"""
Model Training module.
Handles training the Logistic Regression model, displaying evaluation metrics, 
and allowing hyperparameter tuning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from utils import friendly_name, render_footer

def render(clean_df):
    st.markdown("# Logistic Regression Model")
    
    st.markdown(
        '<div class="info-box">'
        "A <strong>Logistic Regression</strong> algorithm with an L1 (Lasso) penalty is used to "
        "predict the probability of a user enrolling. The L1 regularization naturally performs "
        "feature selection by pushing the coefficients of weaker predictors to zero.<br><br>"
        "Use the sidebar to adjust training parameters such as the test set size and "
        "cross-validation folds."
        "</div>",
        unsafe_allow_html=True,
    )

    # Configuration sidebar for the model
    with st.sidebar:
        st.markdown("### Model Settings")
        test_size = st.slider(
            "Test set size",
            0.10, 0.40, 0.20, 0.05,
            help="Proportion of the dataset to withhold for evaluation.",
        )
        cv_folds = st.slider(
            "Cross-validation folds",
            3, 15, 10,
            help="Number of folds for cross-validation. Higher values provide a more stable estimate but require more computation time.",
        )
        run_grid = st.checkbox(
            "Run Grid Search tuning",
            value=False,
            help="Automatically search for the optimal regularization strength (C) and penalty type.",
        )

    # Prepare data for training
    model_df = clean_df.copy()
    response = model_df["enrolled"]
    model_df = model_df.drop(columns="enrolled")

    X_train, X_test, y_train, y_test = train_test_split(
        model_df, response, test_size=test_size, random_state=0
    )
    test_ids = X_test["user"]
    
    # Remove the user identifier as it should not be a predictive feature
    X_train = X_train.drop(columns=["user"])
    X_test = X_test.drop(columns=["user"])

    # Scale the features to ensure the regularization is applied evenly
    sc = StandardScaler()
    X_train_sc = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_sc = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Train the model
    clf = LogisticRegression(random_state=0, penalty="l1", solver="liblinear", max_iter=1000)
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)
    y_proba = clf.predict_proba(X_test_sc)[:, 1]

    # Calculate classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Display key metrics
    st.markdown("### Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.4f}", help="Overall correctness of the model's predictions")
    m2.metric("Precision", f"{prec:.4f}", help="Proportion of predicted enrollments that were actually correct")
    m3.metric("Recall", f"{rec:.4f}", help="Proportion of actual enrollments that the model successfully identified")
    m4.metric("F1 Score", f"{f1:.4f}", help="The harmonic mean of precision and recall")

    tab_cm, tab_roc, tab_coef, tab_cv = st.tabs(
        ["Confusion Matrix", "ROC Curve", "Feature Importance", "Cross-Validation"]
    )

    # Confusion matrix visualization
    with tab_cm:
        st.caption(
            "This matrix compares the model's predictions against the actual outcomes. "
            "Correct predictions are located on the diagonal from top-left to bottom-right."
        )
        
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(
            go.Heatmap(
                z=cm,
                x=["Predicted: Not Enrolled", "Predicted: Enrolled"],
                y=["Actual: Not Enrolled", "Actual: Enrolled"],
                colorscale=[[0, "#1a1d29"], [1, "#6C63FF"]],
                text=cm, texttemplate="%{text:,}", textfont={"size": 24},
                showscale=False,
                hovertemplate="<b>%{y}</b> -> <b>%{x}</b><br>Count: %{z:,}<extra></extra>",
            )
        )
        fig_cm.update_layout(
            height=450, 
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve visualization
    with tab_roc:
        st.caption(
            "The Receiver Operating Characteristic (ROC) curve evaluates the trade-off between the "
            "true positive rate and false positive rate. An Area Under Curve (AUC) of 1.0 represents "
            "perfect classification, while 0.5 represents a random guess."
        )
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Model (AUC = {roc_auc:.3f})",
            line=dict(color="#6C63FF", width=3),
            fill="tozeroy", fillcolor="rgba(108,99,255,0.15)",
        ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#555"), name="Random Baseline",
        ))
        
        fig_roc.update_layout(
            height=450, 
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="False Positive Rate", 
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature coefficients visualization
    with tab_coef:
        st.caption(
            "These coefficients reflect each feature's contribution to the prediction. "
            "Positive values increase the likelihood of enrollment, while negative values decrease it. "
            "Features assigned a coefficient of zero were removed by the L1 penalty."
        )
        
        coef_df = pd.DataFrame({
            "Feature": [friendly_name(f) for f in X_train.columns],
            "Coefficient": clf.coef_[0],
        }).sort_values("Coefficient")

        fig_coef = go.Figure(go.Bar(
            x=coef_df["Coefficient"], 
            y=coef_df["Feature"], 
            orientation="h",
            marker=dict(
                color=coef_df["Coefficient"],
                colorscale=[[0, "#FF6584"], [0.5, "#3a3f5c"], [1, "#43E8D8"]],
                showscale=True, 
                colorbar=dict(title="Coefficient"),
            ),
            hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>",
        ))
        
        fig_coef.update_layout(
            height=max(480, len(coef_df) * 24), 
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Coefficient Value",
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    # Cross-validation results
    with tab_cv:
        st.caption(
            "Cross-validation tests the model's reliability by training and evaluating it across "
            "different segments of the dataset. Consistent scores imply a robust model."
        )
        
        cv_scores = cross_val_score(clf, X_train_sc, y_train, cv=cv_folds)
        
        st.success(
            f"**Mean Accuracy:** {cv_scores.mean():.4f} +/- {cv_scores.std() * 2:.4f} "
            f"(95% confidence interval)"
        )

        fig_cv = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(cv_folds)],
            y=cv_scores, 
            marker_color="#6C63FF",
            text=[f"{s:.4f}" for s in cv_scores], 
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>",
        ))
        
        fig_cv.add_hline(
            y=cv_scores.mean(), 
            line_dash="dash", 
            line_color="#FF6584",
            annotation_text=f"Mean = {cv_scores.mean():.4f}"
        )
        
        fig_cv.update_layout(
            height=400, 
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Accuracy",
            yaxis_range=[max(0, cv_scores.min() - 0.05), min(1, cv_scores.max() + 0.05)],
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    # Grid Search logic
    if run_grid:
        st.markdown("---")
        st.markdown("### Hyperparameter Tuning")
        st.caption(
            "Performing an exhaustive search over a grid of hyperparameters to locate the optimal configuration."
        )
        
        with st.spinner("Executing Grid Search... this process may take a few moments."):
            param_grid = {
                "C": [0.001, 0.01, 0.1, 0.5, 0.9, 1, 2, 5, 10, 100],
                "penalty": ["l1", "l2"],
            }
            gs = GridSearchCV(
                LogisticRegression(random_state=0, solver="liblinear", max_iter=1000),
                param_grid, 
                scoring="accuracy", 
                cv=cv_folds, 
                n_jobs=-1,
            )
            
            t0 = time.time()
            gs.fit(X_train_sc, y_train)
            elapsed_time = time.time() - t0

        col_g1, col_g2, col_g3 = st.columns(3)
        col_g1.metric("Best Accuracy", f"{gs.best_score_:.4f}")
        col_g2.metric("Optimal C Parameter", gs.best_params_["C"])
        col_g3.metric("Computation Time", f"{elapsed_time:.1f}s")
        
        with st.expander("Review optimal parameters"):
            st.json(gs.best_params_)

    st.markdown("---")
    
    # Table of sample predictions
    st.markdown("### Sample Predictions vs Actuals")
    st.caption("A subset of predictions from the test set compared directly against their true outcomes.")
    
    results = pd.DataFrame({
        "User ID": test_ids.values,
        "Actual Status": y_test.values.astype(int),
        "Predicted Status": y_pred.astype(int),
        "Confidence (%)": np.round(y_proba * 100, 1),
    }).reset_index(drop=True)

    def format_status(val):
        """Format table cells based on their status."""
        return "color: #43E8D8; font-weight: 600" if val == 1 else "color: #FF6584"

    st.dataframe(
        results.head(50).style.applymap(format_status, subset=["Actual Status", "Predicted Status"]),
        use_container_width=True, 
        height=400,
    )

    render_footer()
