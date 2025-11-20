# Google Cloud MLOps Pipeline

An end-to-end machine learning operations (MLOps) pipeline built with Google Cloud Vertex AI, demonstrating the complete ML lifecycle from data preparation to model deployment and monitoring.

## 🚀 Quick Start

### Prerequisites
- Google Cloud Project with billing enabled
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed
- Python 3.8+

### Setup (5 minutes)

1. **Clone and navigate to project**:
   ```bash
   git clone <your-repo-url>
   cd GCP_MLOps
   ```

2. **Set your Google Cloud Project ID**:
   ```bash
   export GCP_PROJECT_ID="your-actual-project-id"
   ```

3. **Run the automated setup script**:
   ```bash
   ./setup_gcp.sh
   ```

4. **Verify setup**:
   ```bash
   python verify_setup.py
   ```

5. **Start exploring**:
   ```bash
   jupyter lab
   # Open notebooks/01_getting_started.ipynb
   ```

That's it! 🎉 Your Google Cloud MLOps pipeline is ready.

## 📊 Project Overview

This project implements a production-ready MLOps pipeline using Google Cloud Vertex AI services, featuring:

- **🔄 Complete MLOps Pipeline**: End-to-end workflow from data → training → deployment
- **☁️ Cloud-Native Architecture**: Built specifically for Google Cloud Platform and Vertex AI
- **🚀 5-Minute Setup**: Automated deployment script with comprehensive validation
- **📊 Advanced Data Processing**: Validation, quality checks, feature engineering
- **🤖 Multi-Algorithm Training**: 7 ML algorithms with hyperparameter tuning
- **📦 Artifact Management**: Model versioning and cloud storage integration
- **📓 Interactive Learning**: Step-by-step Jupyter notebook tutorials
- **💰 Cost Optimized**: Designed for learning with minimal cloud costs (~$6-27/month)

## 🏗️ Architecture

### Core Components

1. **📊 Data Pipeline** - Automated loading, validation, and preprocessing
2. **🤖 Model Training** - Multi-algorithm training with hyperparameter optimization
3. **☁️ Cloud Integration** - Vertex AI training jobs and model registry
4. **🚀 Deployment** - Automated endpoint deployment and serving
5. **🔄 Orchestration** - End-to-end pipeline automation
6. **📈 Monitoring** - Performance tracking and model monitoring

### Technology Stack
- **Primary**: Google Vertex AI, Cloud Storage, Artifact Registry
- **SDK**: Vertex AI Python SDK, Google Cloud SDK  
- **ML**: Scikit-learn, TensorFlow, Joblib
- **Development**: Jupyter Lab, Python 3.8+
- **Orchestration**: Vertex AI Pipelines (KFP)

## 🎯 Quick Setup Summary

Your project includes:
✅ **8 Python modules** - Complete MLOps framework  
✅ **6 Jupyter notebooks** - Interactive learning journey  
✅ **Automated setup script** - One-command deployment  
✅ **Verification tools** - Comprehensive testing  
✅ **Documentation** - Detailed guides and examples  
✅ **Cost optimization** - Efficient resource management  

**Total Setup Time**: 5 minutes  
**Learning Time**: 3-5 hours for complete mastery  
**Monthly Cost**: $6-27 for development use

2. **Model Training** (`notebooks/03_model_training.ipynb`)
   - Local multi-algorithm training with hyperparameter tuning
   - Comprehensive model evaluation and comparison
   - Model persistence and metadata management

3. **Cloud Training** (`notebooks/04_vertex_ai_training.ipynb`)  
   - Vertex AI Custom Training jobs
   - Cloud-based hyperparameter optimization
   - Model registry integration

4. **Pipeline Orchestration** (`notebooks/05_vertex_ai_pipelines.ipynb`)  
   - Kubeflow Pipelines (KFP) workflows
   - Automated model registry
   - Deployment automation

5. **Model Serving** (`notebooks/06_model_deployment.ipynb`)
   - Real-time inference endpoints
   - Batch prediction jobs
   - Model monitoring

### Technology Stack

- **Primary Platform**: Google Cloud Vertex AI
- **Orchestration**: Vertex AI Pipelines (Kubeflow)
- **Training**: Vertex AI Custom Training, AutoML
- **Serving**: Vertex AI Endpoints, Batch Prediction
- **Storage**: Google Cloud Storage (GCS)
- **Containers**: Google Container Registry / Artifact Registry
- **SDK**: Vertex AI Python SDK (`google-cloud-aiplatform`)

## 🚀 Quick Start

### Prerequisites

1. **Google Cloud Account** with billing enabled
   - $300 free tier credit for new accounts
   - GitHub Student Developer Pack (additional credits)

2. **Required APIs Enabled**:
   - Vertex AI API
   - Cloud Storage API
   - Container Registry API
   - Cloud Build API

### Setup Instructions

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd GCP_MLOps
   ```

2. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Google Cloud Authentication**
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

4. **Start with Phase 1**
   ```bash
   jupyter notebook notebooks/01_getting_started.ipynb
   ```

## � Development Phases

### Phase 1: Environment Setup ✅
- [x] Google Cloud project configuration
- [x] Authentication and API access
- [x] Cloud Storage bucket setup
- [x] Sample dataset preparation
- **Notebook**: `01_getting_started.ipynb`

### Phase 2: Data Pipeline 🔄
- [ ] Data preprocessing and validation
- [ ] Feature engineering pipelines
- [ ] Data quality checks
- **Notebook**: `02_data_processing_pipeline.ipynb`

### Phase 3: Model Training 🔄
- [ ] Vertex AI Custom Training jobs
- [ ] Hyperparameter optimization
- [ ] Model evaluation and metrics
- **Notebook**: `03_model_training.ipynb`

### Phase 4: Pipeline Orchestration 🔄  
- [ ] Kubeflow Pipelines implementation
- [ ] Model registry integration
- [ ] Automated workflows
- **Notebook**: `04_vertex_ai_pipelines.ipynb`

### Phase 5: Model Deployment 🔄
- [ ] Real-time inference endpoints
- [ ] Batch prediction setup
- [ ] Model monitoring
- **Notebook**: `05_model_deployment.ipynb`

## 🗂️ Project Structure

```
GCP_MLOps/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── utils.py                  # Utility functions
│   ├── data/                     # Data processing modules
│   ├── models/                   # Model training modules
│   └── pipelines/                # Pipeline components
├── notebooks/                    # Jupyter notebooks for development
│   ├── 01_getting_started.ipynb
│   ├── 02_data_processing_pipeline.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_vertex_ai_pipelines.ipynb
│   └── 05_model_deployment.ipynb
├── configs/                      # Configuration files
│   └── config.yaml
├── data/                        # Local data storage
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── PLANNING.md                  # Detailed project plan
└── README.md                    # This file
```

## ⚙️ Configuration

Update `configs/config.yaml` with your project settings:

```yaml
gcp:
  project_id: "your-project-id"
  region: "us-central1"
  
storage:
  bucket_name: "your-bucket-name"
  
vertex_ai:
  location: "us-central1"
```

## 💰 Cost Management

This project is designed to minimize GCP costs:

- **Free Tier**: Utilizes $300 free credit
- **Small Datasets**: Uses lightweight Iris dataset (< 10KB)
- **Optimized Resources**: Smallest viable machine types
- **Auto-cleanup**: Scripts to delete resources after use

**Estimated costs**: < $10 for complete pipeline development

## � Learning Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Kubeflow Pipelines Guide](https://www.kubeflow.org/docs/components/pipelines/)
- [MLOps Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the established patterns
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Check `PLANNING.md` for detailed implementation strategy
- Review individual notebook documentation
- Open issues for bugs or questions

---

**🎯 Goal**: Build production-ready MLOps skills with Google Cloud Vertex AI while staying within free tier limits!
