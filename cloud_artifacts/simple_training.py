#!/usr/bin/env python3
import pandas as pd
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse

def train_model():
    """Simple training function for Vertex AI"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='random_forest', 
                       help='Type of model to train')
    args = parser.parse_args()

    print(f"🚀 Starting Vertex AI training: {args.model_type}")

    # Use the iris dataset directly (since we uploaded it)
    # In a real scenario, you'd load from GCS
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    if args.model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif args.model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif args.model_type == 'svm':
        model = SVC(random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    print(f"📚 Training {type(model).__name__}...")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    print(f"✅ Training completed!")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"📊 Precision: {precision:.4f}")
    print(f"📊 Recall: {recall:.4f}")
    print(f"📊 F1-Score: {f1:.4f}")

    # Save results (simulated - in real scenario would save to GCS)
    results = {
        'model_type': args.model_type,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'status': 'SUCCESS'
    }

    print(f"💾 Results: {results}")
    print("🎯 Training job completed successfully!")

    return results

if __name__ == "__main__":
    train_model()
