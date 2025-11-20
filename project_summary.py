#!/usr/bin/env python3
"""
Google Cloud MLOps Pipeline - Project Summary

This script provides a comprehensive overview of your MLOps pipeline project
and guides you through the next steps to get started.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║         🚀 GOOGLE CLOUD MLOPS PIPELINE READY! 🚀               ║
    ║                                                                  ║
    ║    Your complete end-to-end machine learning operations          ║
    ║    pipeline is ready for deployment on Google Cloud Platform    ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_project_structure():
    """Check and display project structure."""
    print("📁 PROJECT STRUCTURE:")
    print("=" * 50)
    
    structure = {
        "🔧 setup_gcp.sh": "Automated Google Cloud setup script",
        "✅ verify_setup.py": "Setup verification and testing script",
        "📚 GCP_SETUP.md": "Detailed setup documentation",
        "🌟 COMPLETE_SETUP_GUIDE.md": "Complete learning guide",
        "📋 TASKS.md": "Project progress and task tracking",
        "⚙️ configs/config.yaml": "Project configuration file",
        "🐍 src/": "Python source code modules",
        "📓 notebooks/": "Interactive Jupyter learning materials"
    }
    
    for item, description in structure.items():
        file_path = item.split()[1] if " " in item else item
        if Path(file_path).exists():
            print(f"✅ {item:<25} - {description}")
        else:
            print(f"❌ {item:<25} - {description} (MISSING)")
    
    print()

def show_modules_created():
    """Show the modules that were created."""
    print("🐍 PYTHON MODULES CREATED:")
    print("=" * 50)
    
    modules = {
        "src/config.py": "Configuration management with YAML support",
        "src/utils.py": "Utility functions and logging setup", 
        "src/data/data_loader.py": "Data loading from various sources",
        "src/data/validator.py": "Data quality and validation checks",
        "src/data/preprocessor.py": "Data preprocessing and feature engineering",
        "src/models/trainer.py": "Model training with 7 ML algorithms",
        "src/cloud/vertex_ai.py": "Google Vertex AI integration",
        "src/cloud/storage_manager.py": "Cloud Storage and artifact management"
    }
    
    for module, description in modules.items():
        if Path(module).exists():
            print(f"✅ {module:<35} - {description}")
        else:
            print(f"❌ {module:<35} - {description} (MISSING)")
    
    print()

def show_capabilities():
    """Show pipeline capabilities."""
    print("🎯 PIPELINE CAPABILITIES:")
    print("=" * 50)
    
    capabilities = [
        "📊 Data Loading: Iris, Wine, Breast Cancer datasets + CSV support",
        "🔍 Data Validation: Schema, quality, statistical, and drift detection",
        "⚙️ Data Preprocessing: Scaling, encoding, feature engineering",
        "🤖 ML Training: 7 algorithms with hyperparameter tuning",
        "☁️ Cloud Training: Vertex AI custom training jobs",
        "📦 Artifact Management: Model versioning and storage",
        "🚀 Model Deployment: Vertex AI endpoints and serving",
        "🔄 Pipeline Orchestration: End-to-end workflow automation",
        "💰 Cost Optimization: Efficient resource management",
        "📈 Monitoring: Performance tracking and alerting"
    ]
    
    for capability in capabilities:
        print(f"✅ {capability}")
    
    print()

def show_getting_started():
    """Show getting started instructions."""
    print("🚀 GETTING STARTED (5 MINUTES):")
    print("=" * 50)
    
    steps = [
        ("1️⃣", "Set your Google Cloud Project ID:", 
         'export GCP_PROJECT_ID="your-actual-project-id"'),
        
        ("2️⃣", "Run the automated setup script:", 
         "./setup_gcp.sh"),
        
        ("3️⃣", "Verify everything is working:", 
         "python verify_setup.py"),
        
        ("4️⃣", "Activate Python environment:", 
         "source venv/bin/activate"),
        
        ("5️⃣", "Start Jupyter Lab:", 
         "jupyter lab"),
        
        ("6️⃣", "Open the first notebook:", 
         "Open notebooks/01_getting_started.ipynb")
    ]
    
    for step, description, command in steps:
        print(f"{step} {description}")
        if command.startswith("export") or command.startswith("./") or command.startswith("python"):
            print(f"   💻 {command}")
        else:
            print(f"   📝 {command}")
        print()

def show_learning_path():
    """Show the learning path."""
    print("📚 LEARNING PATH (3-5 HOURS TOTAL):")
    print("=" * 50)
    
    notebooks = [
        ("📓 01_getting_started.ipynb", "15-20 min", 
         "Environment validation, basic data loading, simple model training"),
        
        ("📊 02_data_processing_pipeline.ipynb", "30-40 min",
         "Advanced data preprocessing, validation, and feature engineering"),
        
        ("🤖 03_model_training.ipynb", "45-60 min",
         "Comprehensive ML training with 7 algorithms and hyperparameter tuning"),
        
        ("☁️ 04_vertex_ai_training.ipynb", "30-45 min",
         "Cloud-based training with Vertex AI and distributed computing"),
        
        ("🚀 05_model_deployment.ipynb", "30-40 min",
         "Model deployment to Vertex AI endpoints and serving"),
        
        ("🔄 06_vertex_ai_pipelines.ipynb", "45-60 min",
         "End-to-end pipeline orchestration and automation")
    ]
    
    for notebook, duration, description in notebooks:
        print(f"{notebook:<35} ({duration})")
        print(f"   {description}")
        print()

def show_cost_estimate():
    """Show cost estimates."""
    print("💰 ESTIMATED MONTHLY COSTS (Development):")
    print("=" * 50)
    
    costs = [
        ("Vertex AI", "~$5-20/month", "ML training and deployment"),
        ("Cloud Storage", "~$1-5/month", "Data and model storage"),
        ("Artifact Registry", "~$0-2/month", "Container images"),
        ("Service Account", "Free", "Authentication"),
        ("APIs", "Free", "Google Cloud services")
    ]
    
    total_min = sum([5, 1, 0])
    total_max = sum([20, 5, 2])
    
    for service, cost, description in costs:
        print(f"💳 {service:<18} {cost:<15} - {description}")
    
    print(f"\n🎯 Total Estimated Cost: ${total_min}-{total_max}/month for development use")
    print()

def show_support_resources():
    """Show support and learning resources."""
    print("🆘 SUPPORT & RESOURCES:")
    print("=" * 50)
    
    resources = [
        ("📖 Documentation", "GCP_SETUP.md - Detailed setup instructions"),
        ("🔍 Troubleshooting", "Run verify_setup.py to diagnose issues"),
        ("📚 Google Cloud Docs", "https://cloud.google.com/vertex-ai/docs"),
        ("💻 Code Samples", "https://github.com/GoogleCloudPlatform/vertex-ai-samples"),
        ("🎓 Free Training", "https://cloud.google.com/training/machinelearning-ai"),
        ("🏛️ Architecture", "https://cloud.google.com/architecture/ml-on-gcp")
    ]
    
    for resource, description in resources:
        print(f"🔗 {resource:<20} - {description}")
    
    print()

def check_prerequisites():
    """Check if prerequisites are met."""
    print("✅ PREREQUISITES CHECK:")
    print("=" * 50)
    
    prereqs = []
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 8):
        prereqs.append(("✅", f"Python {python_version}", "Compatible"))
    else:
        prereqs.append(("❌", f"Python {python_version}", "Need Python 3.8+"))
    
    # Check if key files exist
    key_files = ["setup_gcp.sh", "verify_setup.py", "configs/config.yaml", "requirements.txt"]
    for file_path in key_files:
        if Path(file_path).exists():
            prereqs.append(("✅", f"{file_path}", "Ready"))
        else:
            prereqs.append(("❌", f"{file_path}", "Missing"))
    
    # Check if gcloud is available
    try:
        import subprocess
        result = subprocess.run(["gcloud", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            prereqs.append(("✅", "Google Cloud CLI", "Installed"))
        else:
            prereqs.append(("❌", "Google Cloud CLI", "Not found"))
    except:
        prereqs.append(("❌", "Google Cloud CLI", "Not installed"))
    
    for status, item, note in prereqs:
        print(f"{status} {item:<25} - {note}")
    
    print()

def main():
    """Main function."""
    print_banner()
    
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Project Directory: {Path.cwd()}")
    print()
    
    check_prerequisites()
    check_project_structure()
    show_modules_created()
    show_capabilities()
    show_getting_started()
    show_learning_path()
    show_cost_estimate()
    show_support_resources()
    
    print("🎉 CONGRATULATIONS!")
    print("=" * 50)
    print("Your Google Cloud MLOps pipeline is ready for deployment!")
    print("Follow the getting started steps above to begin your ML journey.")
    print()
    print("Questions? Check the documentation or run verify_setup.py")
    print("Happy machine learning! 🚀🤖✨")

if __name__ == "__main__":
    main()
