#!/bin/bash

# MLOps Project Setup Script
# Run this script to set up your development environment

echo "🚀 Setting up MLOps project with Vertex AI..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Check if gcloud is installed
if command -v gcloud &> /dev/null; then
    echo "✅ Google Cloud CLI detected"
    gcloud version
else
    echo "⚠️  Google Cloud CLI not found"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
fi

echo ""
echo "🎉 Setup complete! Next steps:"
echo "1. Activate your virtual environment: source venv/bin/activate"
echo "2. Install Google Cloud CLI if not already installed"
echo "3. Follow Task 1.1 in TASKS.md to set up your GCP account"
echo "4. Run: gcloud auth login"
echo "5. Create a new GCP project: gcloud projects create your-project-id"
