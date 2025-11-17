# Cloud MLOps - Google Vertex AI Pipeline Demo

## Overview
This project builds an end-to-end Google Vertex AI MLOps pipeline demonstrating the complete machine learning lifecycle from data annotation to model deployment and endpoint creation. The goal is to create a simplified, local development-friendly implementation that showcases Vertex AI's core MLOps capabilities.

## Project Scope
- **End-to-End ML Pipeline**: Data preparation → Training → Model evaluation → Deployment → Monitoring
- **Local Development Focus**: Minimal cloud costs, development-oriented setup
- **Educational Purpose**: Clear demonstration of Vertex AI MLOps best practices
- **Simple Implementation**: No advanced features, focus on core workflow

## Technical Architecture

### Core Components
1. **Data Pipeline**
   - Data annotation/labeling workflow with Vertex AI Data Labeling
   - Data validation and preprocessing
   - Feature engineering with Feature Store

2. **Training Pipeline**
   - Custom training jobs with Vertex AI Training
   - Hyperparameter tuning with Vertex AI Vizier
   - Model evaluation and validation

3. **Deployment Pipeline**
   - Model Registry for version management
   - Automated endpoint deployment
   - Basic monitoring with Model Monitoring

4. **Orchestration**
   - Vertex AI Pipelines (Kubeflow-based) for workflow automation
   - CI/CD integration with Cloud Build

### Technology Stack
- **Primary**: Google Vertex AI (Pipelines, Training, Endpoints, Model Registry)
- **SDK**: Vertex AI Python SDK, Google Cloud SDK
- **Development**: Jupyter Notebooks, Vertex AI Workbench, Python 3.8+
- **Storage**: Google Cloud Storage (GCS)
- **Orchestration**: Vertex AI Pipelines (KFP)
- **Monitoring**: Vertex AI Model Monitoring
- **Infrastructure**: gcloud CLI, IAM service accounts

### Sample Use Case
**Image Classification Pipeline** (keeping it simple)
- Dataset: Small public image dataset (e.g., CIFAR-10 subset)
- Model: Pre-trained CNN with transfer learning
- Deployment: Real-time inference endpoint

## Development Approach

### Phase 1: Foundation
- Google Cloud project setup and IAM configuration
- Vertex AI Workbench environment
- GCS bucket configuration
- Basic data pipeline

### Phase 2: Core Pipeline
- Data preprocessing and feature engineering
- Custom training pipeline with Vertex AI Training
- Basic evaluation metrics with Vertex AI TensorBoard

### Phase 3: MLOps Integration
- Vertex AI Pipelines implementation (KFP)
- Model Registry integration
- Automated deployment

### Phase 4: Monitoring & Operations
- Vertex AI Model Monitoring setup
- Endpoint management
- Simple CI/CD workflow with Cloud Build

## Success Criteria
- [ ] Complete end-to-end pipeline execution
- [ ] Automated model deployment to Vertex AI Endpoint
- [ ] Working inference endpoint with sample predictions
- [ ] Basic model monitoring dashboard
- [ ] Documented pipeline steps and configuration
- [ ] Cost-effective development setup (minimal GCP charges)

## Constraints & Considerations
- **Budget**: Minimize GCP costs for local development
- **Complexity**: Keep implementation simple and educational
- **Time**: Focus on core MLOps concepts, avoid advanced features
- **Data**: Use small, public datasets to avoid data management complexity
- **Security**: Basic IAM setup, no advanced security features initially

## Learning Objectives
- Understand Vertex AI Pipelines (KFP) workflow orchestration
- Experience with Vertex AI Training and Prediction
- Learn MLOps best practices with Google Cloud tools
- Gain hands-on experience with model lifecycle management
