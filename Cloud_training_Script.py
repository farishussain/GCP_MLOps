# --- TRAINING SCRIPT: Runs once inside the cloud container ---

import joblib
import argparse
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# The saved model file name
MODEL_FILENAME = 'model.joblib'

def train_and_upload_model(model_dir):
    """
    Trains the model and saves it to the specified output directory (GCS path).
    """
    print(f"Starting cloud model training. Output will go to: {model_dir}")
    
    # 1. Load & Split Data 
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train the Model
    model = LogisticRegression(max_iter=200).fit(X_train, y_train)
    
    # 3. Save the Model Artifact to the specified path
    # This path is where Vertex AI mounts the cloud storage bucket
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    joblib.dump(model, model_path)
    
    print("-" * 30)
    print(f"SUCCESS: Model saved to cloud path: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Vertex AI automatically passes the model output directory via this argument
    parser.add_argument(
        '--model-dir', 
        type=str, 
        default=os.environ.get('AIP_MODEL_DIR', './'), 
        help='The directory where the model artifact should be saved (GCS path).'
    )
    args = parser.parse_args()
    
    train_and_upload_model(args.model_dir)
