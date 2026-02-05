
import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the path to the model and preprocessor files
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model and preprocessors
with open(os.path.join(MODEL_DIR, 'log_reg_model.pkl'), 'rb') as f:
    log_reg_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'numerical_features_for_scaling.pkl'), 'rb') as f:
    numerical_features_for_scaling = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'categorical_features_for_encoding.pkl'), 'rb') as f:
    categorical_features_for_encoding = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'X_train_columns.pkl'), 'rb') as f:
    X_train_columns = pickle.load(f)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df_input = pd.DataFrame(data, index=[0])

        # Feature Engineering (re-create day_of_week and hour_of_day)
        df_input['Date'] = pd.to_datetime(df_input['Date'])
        df_input['day_of_week'] = df_input['Date'].dt.day_name()
        df_input['hour_of_day'] = pd.to_datetime(df_input['Time'], format='%H:%M').dt.hour
        
        # Drop columns not used for prediction (as they were removed during training)
        # 'Violation_ID', 'Officer_ID', 'Comments', 'Fine_Amount', 'Date', 'Time'
        cols_to_drop = ['Violation_ID', 'Officer_ID', 'Comments', 'Date', 'Time'] # Fine_Amount was dropped, not here
        df_input = df_input.drop(columns=[col for col in cols_to_drop if col in df_input.columns], errors='ignore')

        # Align columns with X_train_columns. This is crucial for consistent prediction.
        # One-hot encode categorical features
        df_input_encoded = pd.get_dummies(df_input[categorical_features_for_encoding], drop_first=True)

        # Scale numerical features
        df_input_scaled_numerical = pd.DataFrame(scaler.transform(df_input[numerical_features_for_scaling]),
                                               columns=numerical_features_for_scaling,
                                               index=df_input.index)
        
        # Combine processed features
        df_processed_input = pd.concat([df_input_scaled_numerical, df_input_encoded], axis=1)

        # Reindex and fill missing columns with 0 to match training data columns
        df_final = pd.DataFrame(columns=X_train_columns)
        df_final = pd.concat([df_final, df_processed_input], ignore_index=True)
        df_final = df_final.fillna(0)
        df_final = df_final[X_train_columns] # Ensure column order is correct

        prediction = log_reg_model.predict(df_final)
        prediction_proba = log_reg_model.predict_proba(df_final)[:, 1]

        return jsonify({
            'prediction': int(prediction[0]),
            'probability_high_fine': float(prediction_proba[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local development purposes. Gunicorn or similar WSGI server should be used in production.
    app.run(host='0.0.0.0', port=5000, debug=True)
