# --- INFERENCE SERVER: Runs 24/7 on the live API Endpoint ---

import joblib
import json
import numpy as np
from flask import Flask, request, jsonify
import os

# --- Configuration ---
MODEL_FILENAME = 'model.joblib'
app = Flask(__name__)
model = None # Initialize model as None

# --- Load Model (Executed when the server starts) ---
try:
    # Look for the model file in the expected deployment directory
    model_path = os.path.join(os.environ.get('AIP_STORAGE_URI', '.'), MODEL_FILENAME)
    # If using custom containers, model artifacts are often copied locally or loaded from GCS URI
    if os.path.exists(MODEL_FILENAME):
         model = joblib.load(MODEL_FILENAME)
    else:
         # Fallback for custom artifact loading path
         model = joblib.load(model_path)
    print(f"SUCCESS: Model loaded successfully for inference.")
except Exception as e:
    print(f"ERROR: Failed to load model file. Error: {e}")


# --- Define the API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """Confirms the server and model are running."""
    if model is not None:
        return jsonify({'status': 'ok'}), 200
    return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503

@app.route('/predict', methods=['POST'])
def predict_data():
    """Receives new data and returns a classification prediction."""
    if model is None:
        return jsonify({'error': 'Model unavailable.'}), 503

    try:
        data = request.get_json(force=True)
        # Input format is expected to be {"instances": [[5.1, 3.5, 1.4, 0.2], ...]}
        input_data = data.get('instances', [])
        
        features = np.array(input_data)
        predictions = model.predict(features).tolist()
        
        return jsonify({
            'predictions': predictions,
            'species_key': ['Setosa (0)', 'Versicolor (1)', 'Virginica (2)']
        }), 200

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

# --- Start the Flask Server ---
if __name__ == '__main__':
    # Use the port assigned by the Vertex AI environment
    port = int(os.environ.get('AIP_HTTP_PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
