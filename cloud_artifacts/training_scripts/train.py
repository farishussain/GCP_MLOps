#!/usr/bin/env python3
"""
Vertex AI Training Script for Iris Classification

This script trains machine learning models on the Iris dataset using scikit-learn
and is designed to run on Vertex AI custom training jobs.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> tuple:
    """Load and prepare the iris dataset."""
    logger.info(f"Loading data from {data_path}")
    
    if data_path.startswith('gs://'):
        # Handle GCS paths
        import subprocess
        local_path = '/tmp/iris_data.csv'
        subprocess.run(['gsutil', 'cp', data_path, local_path], check=True)
        data_path = local_path
    
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'target']
    X = df[feature_columns].values
    y = df['target'].values
    
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_columns

def train_model(X, y, model_type: str, enable_tuning: bool = False) -> dict:
    """Train a machine learning model."""
    logger.info(f"Training {model_type} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Select model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'svm':
        model = SVC(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    results = {
        'model_type': model_type,
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    logger.info(f"Model trained - Test Accuracy: {test_score:.4f}")
    return model, results

def save_artifacts(model, results: dict, output_path: str):
    """Save model and results."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = str(output_path).replace('.pkl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Upload to GCS if output is GCS path
    if output_path.startswith('gs://'):
        import subprocess
        subprocess.run(['gsutil', 'cp', model_path, output_path], check=True)
        subprocess.run(['gsutil', 'cp', metrics_path, metrics_path.replace('/tmp/', 'gs://')], check=True)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ML model on Iris dataset')
    parser.add_argument('--model-type', default='random_forest', 
                       choices=['random_forest', 'logistic_regression', 'svm'],
                       help='Type of model to train')
    parser.add_argument('--data-path', required=True, help='Path to training data')
    parser.add_argument('--model-output-path', required=True, help='Output path for trained model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--cross-val-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--enable-tuning', action='store_true', help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    logger.info("Starting Vertex AI training job")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.model_output_path}")
    
    try:
        # Load data
        X, y, feature_columns = load_data(args.data_path)
        
        # Train model
        model, results = train_model(X, y, args.model_type, args.enable_tuning)
        
        # Save artifacts
        save_artifacts(model, results, args.model_output_path)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
