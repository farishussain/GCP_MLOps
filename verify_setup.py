#!/usr/bin/env python3
"""
Google Cloud MLOps Setup Verification Script

This script verifies that your Google Cloud MLOps pipeline is set up correctly.
Run this after completing the setup process to ensure everything works.

Usage:
    python verify_setup.py
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Add src to path for imports
sys.path.insert(0, 'src')

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_environment() -> Tuple[bool, List[str]]:
    """Check Python environment and required packages."""
    logger = logging.getLogger(__name__)
    issues = []
    
    logger.info("🐍 Checking Python environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python version {sys.version_info} is too old. Need Python 3.8+")
    
    # Check required packages - mapping import names to package names
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib',
        'yaml': 'PyYAML'
    }
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            issues.append(f"Required package not found: {package_name}")
    
    return len(issues) == 0, issues

def check_google_cloud_packages() -> Tuple[bool, List[str]]:
    """Check Google Cloud specific packages."""
    logger = logging.getLogger(__name__)
    issues = []
    
    logger.info("☁️ Checking Google Cloud packages...")
    
    cloud_packages = [
        'google.cloud.aiplatform',
        'google.cloud.storage',
        'google.auth',
        'kfp'
    ]
    
    for package in cloud_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Google Cloud package not found: {package}")
    
    return len(issues) == 0, issues

def check_configuration() -> Tuple[bool, List[str]]:
    """Check project configuration."""
    logger = logging.getLogger(__name__)
    issues = []
    
    logger.info("⚙️ Checking configuration...")
    
    try:
        from src.config import get_config
        config = get_config()
        
        # Check required configuration values
        if config.gcp.project_id == "your-gcp-project-id":
            issues.append("GCP project ID not set in configuration")
        
        if config.storage.bucket_name == "your-mlops-bucket":
            issues.append("GCS bucket name not set in configuration")
            
    except Exception as e:
        issues.append(f"Configuration loading failed: {e}")
    
    return len(issues) == 0, issues

def check_environment_variables() -> Tuple[bool, List[str]]:
    """Check required environment variables."""
    logger = logging.getLogger(__name__)
    issues = []
    
    logger.info("🔑 Checking environment variables...")
    
    # Try to load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    required_vars = [
        'GCP_PROJECT_ID',
        'GCS_BUCKET', 
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"Environment variable not set: {var}")
    
    # Check if credentials file exists
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path and not Path(creds_path).exists():
        issues.append(f"Credentials file not found: {creds_path}")
    
    return len(issues) == 0, issues

def check_google_cloud_auth() -> Tuple[bool, List[str]]:
    """Check Google Cloud authentication."""
    logger = logging.getLogger(__name__)
    issues = []
    
    logger.info("🔐 Checking Google Cloud authentication...")
    
    try:
        from google.auth import default
        credentials, project = default()
        
        if not project:
            issues.append("No default project found in credentials")
            
    except Exception as e:
        issues.append(f"Google Cloud authentication failed: {e}")
    
    return len(issues) == 0, issues

def check_cloud_services() -> Tuple[bool, List[str]]:
    """Check access to Google Cloud services."""
    logger = logging.getLogger(__name__)
    issues = []
    
    logger.info("🚀 Checking cloud services access...")
    
    # Check Cloud Storage
    try:
        from google.cloud import storage
        from src.config import get_config
        
        config = get_config()
        client = storage.Client()
        bucket = client.bucket(config.storage.bucket_name)
        
        # Try to list blobs
        list(bucket.list_blobs(max_results=1))
        
    except Exception as e:
        issues.append(f"Cloud Storage access failed: {e}")
    
    # Check Vertex AI
    try:
        from google.cloud import aiplatform
        from src.config import get_config
        
        config = get_config()
        aiplatform.init(
            project=config.gcp.project_id,
            location=config.vertex_ai.location
        )
        
        # Try to list models
        aiplatform.Model.list()
        
    except Exception as e:
        issues.append(f"Vertex AI access failed: {e}")
    
    return len(issues) == 0, issues

def check_data_pipeline() -> Tuple[bool, List[str]]:
    """Check data pipeline functionality."""
    logger = logging.getLogger(__name__)
    issues = []
    
    logger.info("📊 Checking data pipeline...")
    
    try:
        from src.data.data_loader import DataLoader
        
        loader = DataLoader()
        X_train, X_test, y_train, y_test = loader.load_dataset('iris')
        
        if X_train.empty or X_test.empty:
            issues.append("Data loading returned empty datasets")
            
    except Exception as e:
        issues.append(f"Data pipeline test failed: {e}")
    
    return len(issues) == 0, issues

def check_model_training() -> Tuple[bool, List[str]]:
    """Check model training functionality."""
    logger = logging.getLogger(__name__)
    issues = []
    
    logger.info("🤖 Checking model training...")
    
    try:
        from src.models.trainer import ModelTrainer
        from src.data.data_loader import DataLoader
        
        # Load sample data
        loader = DataLoader()
        X_train, X_test, y_train, y_test = loader.load_dataset('iris', test_size=0.3)
        
        # Train a simple model
        trainer = ModelTrainer()
        trainer.train_model('random_forest', X_train, y_train)
        
        # Evaluate model  
        evaluation = trainer.evaluate_model('random_forest', X_test, y_test)
        
        if evaluation['accuracy'] < 0.5:
            issues.append("Model training produced poor results (accuracy < 50%)")
            
    except Exception as e:
        issues.append(f"Model training test failed: {e}")
    
    return len(issues) == 0, issues

def run_verification() -> Dict[str, Tuple[bool, List[str]]]:
    """Run all verification checks."""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Starting Google Cloud MLOps Setup Verification")
    logger.info("=" * 60)
    
    checks = {
        'Python Environment': check_python_environment,
        'Google Cloud Packages': check_google_cloud_packages,
        'Configuration': check_configuration,
        'Environment Variables': check_environment_variables,
        'Google Cloud Auth': check_google_cloud_auth,
        'Cloud Services': check_cloud_services,
        'Data Pipeline': check_data_pipeline,
        'Model Training': check_model_training
    }
    
    results = {}
    
    for check_name, check_func in checks.items():
        try:
            success, issues = check_func()
            results[check_name] = (success, issues)
            
            if success:
                logger.info(f"✅ {check_name}: PASSED")
            else:
                logger.error(f"❌ {check_name}: FAILED")
                for issue in issues:
                    logger.error(f"   - {issue}")
                    
        except Exception as e:
            logger.error(f"❌ {check_name}: ERROR - {e}")
            results[check_name] = (False, [str(e)])
    
    return results

def print_summary(results: Dict[str, Tuple[bool, List[str]]]):
    """Print verification summary."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 60)
    logger.info("📋 VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    total_checks = len(results)
    passed_checks = sum(1 for success, _ in results.values() if success)
    failed_checks = total_checks - passed_checks
    
    logger.info(f"Total Checks: {total_checks}")
    logger.info(f"✅ Passed: {passed_checks}")
    logger.info(f"❌ Failed: {failed_checks}")
    
    if failed_checks == 0:
        logger.info("\n🎉 ALL CHECKS PASSED! Your Google Cloud MLOps pipeline is ready!")
        logger.info("\nNext steps:")
        logger.info("1. Open Jupyter Lab: jupyter lab")
        logger.info("2. Run the notebooks in order:")
        logger.info("   - notebooks/01_getting_started.ipynb")
        logger.info("   - notebooks/02_data_processing_pipeline.ipynb")
        logger.info("   - notebooks/03_model_training.ipynb")
        logger.info("   - notebooks/04_vertex_ai_training.ipynb")
        logger.info("   - notebooks/05_model_deployment.ipynb")
        logger.info("   - notebooks/06_vertex_ai_pipelines.ipynb")
    else:
        logger.error(f"\n⚠️ {failed_checks} CHECKS FAILED!")
        logger.error("Please fix the issues above before proceeding.")
        logger.error("\nTroubleshooting:")
        logger.error("1. Re-run the setup script: ./setup_gcp.sh")
        logger.error("2. Check environment variables: source .env")
        logger.error("3. Verify GCP authentication: gcloud auth list")
        logger.error("4. See GCP_SETUP.md for detailed instructions")

def main():
    """Main function."""
    logger = setup_logging()
    
    try:
        results = run_verification()
        print_summary(results)
        
        # Exit with error code if any checks failed
        failed_checks = sum(1 for success, _ in results.values() if not success)
        sys.exit(0 if failed_checks == 0 else 1)
        
    except KeyboardInterrupt:
        logger.info("\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
