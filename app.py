import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(layout="wide") # For a wider layout

st.title('Breast Cancer Prediction App')

script_dir = os.path.dirname(__file__)

# Load the StandardScaler
scaler = joblib.load(os.path.join(script_dir, 'model', 'scaler.joblib'))

# Load the trained classification models
models = {
    'Logistic Regression': joblib.load(os.path.join(script_dir, 'model', 'logistic_regression_model.joblib')),
    'Decision Tree': joblib.load(os.path.join(script_dir, 'model', 'decision_tree_model.joblib')),
    'K-Nearest Neighbors': joblib.load(os.path.join(script_dir, 'model', 'k-nearest_neighbors_model.joblib')),
    'Gaussian Naive Bayes': joblib.load(os.path.join(script_dir, 'model', 'gaussian_naive_bayes_model.joblib')),
    'Random Forest': joblib.load(os.path.join(script_dir, 'model', 'random_forest_model.joblib')),
    'XGBoost': joblib.load(os.path.join(script_dir, 'model', 'xgboost_model.joblib'))
}
# Load model performance metrics (overall from training/testing split)
model_performance = joblib.load(os.path.join(script_dir, 'model', 'model_performance.joblib'))
performance_df = pd.DataFrame(model_performance).T

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

# Radio button for input method
input_method = st.sidebar.radio(
    "Choose input method:",
    ('Manual Input', 'Upload Dataset (CSV)')
    #  ('Upload Dataset (CSV)')
)

input_df = None
y_true_uploaded = None # Initialize for uploaded CSV ground truth

if input_method == 'Manual Input':
    st.sidebar.subheader('Manual Input Sliders')
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

            data[feature] = st.sidebar.slider(f'Select {feature}', float(min_val), float(max_val), float(default_value), key=feature) # Added key for uniqueness
        return pd.DataFrame([data])
    input_df = user_input_features()

elif input_method == 'Upload Dataset (CSV)':
    st.sidebar.subheader('Upload CSV File')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        temp_df = pd.read_csv(uploaded_file)
        st.subheader('Uploaded CSV Data (first 5 rows)')
        st.write(temp_df.head())

        # Check for 'Diagnosis' column
        if 'Diagnosis' in temp_df.columns:
            st.success("The 'Diagnosis' column was found in the uploaded CSV. Evaluation metrics and a confusion matrix will be generated.")
            le = LabelEncoder()
            y_true_uploaded = le.fit_transform(temp_df['Diagnosis'])
            input_df = temp_df[feature_names] # Use only predefined features for X
        else:
            st.warning("The 'Diagnosis' column was not found in the uploaded CSV. Only predictions will be displayed.")
            # Ensure all required features are present
            missing_features = [f for f in feature_names if f not in temp_df.columns]
            if missing_features:
                st.error(f"Error: The uploaded CSV is missing the following required features: {', '.join(missing_features)}. Please ensure your CSV contains all 30 feature columns.")
                st.stop() # Stop execution if critical features are missing
            input_df = temp_df[feature_names]
    else:
        st.info("Please upload a CSV file to proceed with prediction.")
        st.stop() # Stop execution if no CSV is uploaded

# Only proceed if input_df is not None (i.e., either manual input or valid CSV uploaded)
if input_df is not None:
    st.subheader('Features Used for Prediction')
    st.write(input_df)

    # Scale the input features
    input_scaled = scaler.transform(input_df)

    st.subheader('Prediction Results')

    # Dropdown for model selection
    selected_model_name = st.selectbox("Select a model for prediction:", list(models.keys()))
    selected_model = models[selected_model_name]

    # Make predictions with the selected model
    prediction = selected_model.predict(input_scaled)
    # Ensure predict_proba is available for all models for AUC calculation
    if hasattr(selected_model, 'predict_proba'):
        prediction_proba = selected_model.predict_proba(input_scaled)
    else:
        prediction_proba = np.array([[0.5, 0.5]] * len(prediction)) # Default if proba not available

    st.write(f'**{selected_model_name} Prediction:**')
    # If the input was a single row (manual or single row CSV)
    if input_df.shape[0] == 1:
        if prediction[0] == 0:
            st.success(f'Diagnosis: Benign (Probability: {prediction_proba[0][0]:.2f})')
        else:
            st.error(f'Diagnosis: Malignant (Probability: {prediction_proba[0][1]:.2f})')
    else:
        # If multiple rows from CSV, display predictions for each
        prediction_results = pd.DataFrame({
            'Predicted Diagnosis': np.where(prediction == 0, 'Benign', 'Malignant'),
            'Probability Benign': prediction_proba[:, 0],
            'Probability Malignant': prediction_proba[:, 1]
        })
        st.dataframe(prediction_results)

    # Conditional evaluation metrics and confusion matrix for uploaded CSV with 'Diagnosis' column
    if y_true_uploaded is not None:
        st.subheader(f'Evaluation Metrics for {selected_model_name} on Uploaded Data')
        acc = accuracy_score(y_true_uploaded, prediction)
        prec = precision_score(y_true_uploaded, prediction)
        rec = recall_score(y_true_uploaded, prediction)
        f1 = f1_score(y_true_uploaded, prediction)
        auc = roc_auc_score(y_true_uploaded, prediction_proba[:, 1])
        mcc = matthews_corrcoef(y_true_uploaded, prediction)

        eval_metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC Score', 'MCC Score'],
            'Value': [acc, prec, rec, f1, auc, mcc]
        })
        st.dataframe(eval_metrics_df.set_index('Metric'))

        st.subheader(f'Confusion Matrix for {selected_model_name} on Uploaded Data')
        cm = confusion_matrix(y_true_uploaded, prediction)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {selected_model_name}')
        st.pyplot(plt)
        plt.close() # Close the plot to prevent display issues

    # Display overall model performance metrics (from training/testing split)
    st.subheader('Overall Model Performance Metrics (from Training/Testing)')
    st.dataframe(performance_df)

    st.subheader('Overall Performance Comparison Chart')
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC Score', 'MCC Score']
    st.bar_chart(performance_df[metrics_to_plot])
