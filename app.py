import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title('Breast Cancer Prediction App')

# Load the StandardScaler
scaler = joblib.load('scaler.joblib')

# Load the trained classification models
models = {
    'Logistic Regression': joblib.load('logistic_regression_model.joblib'),
    'Decision Tree': joblib.load('decision_tree_model.joblib'),
    'K-Nearest Neighbors': joblib.load('k-nearest_neighbors_model.joblib'),
    'Gaussian Naive Bayes': joblib.load('gaussian_naive_bayes_model.joblib'),
    'Random Forest': joblib.load('random_forest_model.joblib'),
    'XGBoost': joblib.load('xgboost_model.joblib')
}

# Define feature names from the original X DataFrame for input fields
feature_names = [
    'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1',
    'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
    'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2',
    'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2',
    'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3',
    'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3'
]

st.sidebar.header('Input Features')

def user_input_features():
    data = {}
    for feature in feature_names:
        # More specific defaults/ranges for demonstration based on a general understanding of the dataset
        if 'radius' in feature:
            min_val, max_val, default_value = 5.0, 30.0, 15.0
        elif 'texture' in feature:
            min_val, max_val, default_value = 5.0, 40.0, 20.0
        elif 'perimeter' in feature:
            min_val, max_val, default_value = 40.0, 200.0, 100.0
        elif 'area' in feature:
            min_val, max_val, default_value = 100.0, 2500.0, 800.0
        elif 'smoothness' in feature:
            min_val, max_val, default_value = 0.05, 0.15, 0.1
        elif 'compactness' in feature:
            min_val, max_val, default_value = 0.0, 0.3, 0.1
        elif 'concavity' in feature:
            min_val, max_val, default_value = 0.0, 0.5, 0.2
        elif 'concave_points' in feature:
            min_val, max_val, default_value = 0.0, 0.2, 0.08
        elif 'symmetry' in feature:
            min_val, max_val, default_value = 0.1, 0.3, 0.2
        elif 'fractal_dimension' in feature:
            min_val, max_val, default_value = 0.05, 0.1, 0.06
        else:
            min_val, max_val, default_value = 0.0, 100.0, 0.5 # Fallback for any unhandled features

        data[feature] = st.sidebar.slider(f'Select {feature}', float(min_val), float(max_val), float(default_value))
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

# Scale the input features
input_scaled = scaler.transform(input_df)

st.subheader('Prediction Results')

# Make predictions with each model
for name, model in models.items():
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.write(f'**{name} Prediction:**')
    if prediction[0] == 0:
        st.success(f'Diagnosis: Benign (Probability: {prediction_proba[0][0]:.2f})')
    else:
        st.error(f'Diagnosis: Malignant (Probability: {prediction_proba[0][1]:.2f})')

# Load model performance metrics
model_performance = joblib.load('model_performance.joblib')
performance_df = pd.DataFrame(model_performance).T

st.subheader('Model Performance Metrics')
st.dataframe(performance_df)

st.subheader('Performance Comparison Chart')
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC Score', 'MCC Score']
st.bar_chart(performance_df[metrics_to_plot])
