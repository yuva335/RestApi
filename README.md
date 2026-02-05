# Traffic Violation Fine Prediction REST API

This project builds a Machine Learning model to predict whether a traffic violation results in a high fine.
The trained Logistic Regression model is deployed as a Flask REST API.

This project was developed in Google Colab and converted into a REST API using structured prompt-based assistance with Google Gemini.

Project Objective

To analyze the Indian_Traffic_Violations.csv dataset and:

Perform data preprocessing and feature engineering

Create a binary classification target (high_fine)

Train and evaluate a Logistic Regression model

Convert the trained model into a deployable REST API

Prepare the API for cloud deployment (Render compatible)

Development Environment

Platform: Google Colab

AI Assistance: Google Gemini (prompt-based code generation)

Language: Python

Framework: Flask

Deployment Server: Gunicorn

Hosting Target: Render

 Machine Learning Workflow
1️ Data Preprocessing

Loaded dataset: Indian_Traffic_Violations.csv

Created target variable:

high_fine = 1 if Fine_Amount > 1000
high_fine = 0 otherwise


Applied:

Categorical feature encoding

Numerical feature scaling

Train-Test split (80/20)

2️ Model Training

Model Used: Logistic Regression

Evaluation Metrics:

Accuracy

Recall

ROC-AUC

All preprocessing artifacts were saved to ensure consistent prediction during API usage.

 Saved Model Artifacts

The following files are used by the API:

log_reg_model.pkl
scaler.pkl
X_train_columns.pkl
categorical_features_for_encoding.pkl
numerical_features_for_scaling.pkl


These files ensure proper feature alignment and scaling at prediction time.

 REST API Structure
Base Endpoint
/


Returns confirmation that the API is running.

Prediction Endpoint
POST /predict

Example Request Body:
{
  "features": [value1, value2, value3, ...]
}

Example Response:
{
  "prediction": 1
}


Where:

1 → High Fine

0 → Low Fine

 Deployment Configuration

For deployment on Render:

Build Command
pip install -r requirements.txt

Start Command
gunicorn app:app

The Flask application instance is named app inside app.py, which is compatible with Gunicorn deployment.
