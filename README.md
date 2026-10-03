# App Subscription Predictor

**Author**: Saksham Sahu

This project is a Streamlit dashboard that analyzes how users navigate through a mobile application and predicts whether they will subscribe (enroll) to the premium service. It utilizes a Logistic Regression model with L1 regularization to identify the in-app behaviors that serve as the strongest signals for user conversion.

## Features

The dashboard is structured into five modular sections:

1. **Overview**: An introduction to the project goals, high-level dataset metrics, a preview of the raw data, and a summary chart of enrollment distributions.
2. **Explore Data**: Exploratory Data Analysis (EDA) section containing feature distribution histograms, correlation bar charts, and a complete correlation heatmap.
3. **Feature Engineering**: A step-by-step breakdown of the data cleaning pipeline. This includes details regarding the 48-hour cutoff rule and how individual screens are aggregated into functional funnels.
4. **Train Model**: A comprehensive view of the Logistic Regression model. This tab displays performance metrics (Accuracy, Precision, Recall, F1 Score), a confusion matrix, an ROC curve, feature importance rankings, cross-validation results, and an interactive hyperparameter tuning tool via Grid Search.
5. **Predict User**: An interactive interface that allows you to simulate a hypothetical user profile and receive a real-time prediction regarding their likelihood to subscribe.

## Project Structure

```text
streamlit_app/
├── app.py                  # Main entry point and routing handler for the Streamlit application
├── utils.py                # Shared utilities, including data loading and feature engineering logic
├── page_overview.py        # Logic and layout for the Overview tab
├── page_explore.py         # Logic and layout for the Explore Data tab
├── page_features.py        # Logic and layout for the Feature Engineering tab
├── page_model.py           # Logic and layout for the Train Model tab
├── page_predict.py         # Logic and layout for the Predict User tab
├── requirements.txt        # Required Python packages and dependencies
├── appdata10.csv           # Raw user session dataset
├── top_screens.csv         # Reference data detailing the most visited application screens
└── .streamlit/
    └── config.toml         # Custom configuration for the Streamlit UI theme
```

## How to Run Locally

1. **Install dependencies**:
   Ensure you have Python installed, then run the following command to install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit application**:
   Start the local server by running:
   ```bash
   streamlit run app.py
   ```

3. **View the dashboard**:
   Open a web browser and navigate to the local URL provided in your terminal (typically `http://localhost:8501`).
