# Cloud MLOps - Initial Tasks (Vertex AI)

## Phase 1: Environment Setup & Foundation (Week 1)

### Task 1.1: Student Access & Free Tier Setup
- [ ] Apply for Google Cloud for Education (if available at your institution)
- [x] Sign up for Google Cloud Free Tier ($300 credit for new accounts) ✅ **Completed 2025-11-18**
- [ ] Apply for GitHub Student Developer Pack (includes GCP credits)
- [x] Set up billing alerts to monitor usage and avoid unexpected charges ✅ **Completed 2025-11-18**
- [x] Review GCP Always Free tier limits for ongoing free usage ✅ **Completed 2025-11-18**

### Task 1.2: Google Cloud Environment Setup
- [x] Create/configure Google Cloud project with billing enabled ✅ **Completed 2025-11-18** (Project: mlops-295610)
- [x] Set up IAM service account with Vertex AI and GCS permissions ✅ **Completed 2025-11-18**
- [x] Install and configure Google Cloud CLI (gcloud) locally ✅ **Completed 2025-11-18**
- [x] Enable required APIs (Vertex AI, Cloud Storage, Cloud Build) ✅ **Completed 2025-11-18**
- [x] Test GCP connectivity and permissions ✅ **Completed 2025-11-18**

### Task 1.3: Development Environment
- [x] Install Python 3.8+ and create virtual environment ✅ **Completed 2025-11-18**
- [x] Install Vertex AI Python SDK and dependencies ✅ **Completed 2025-11-18**
- [ ] Set up Vertex AI Workbench instance (optional for cloud development)
- [x] Install Jupyter Notebook/Lab for local development ✅ **Completed 2025-11-18**
- [x] Set up VS Code with Google Cloud and Python extensions ✅ **Completed 2025-11-18**

### Task 1.4: Cloud Storage Setup
- [x] Create GCS bucket for project data and artifacts ✅ **Completed 2025-11-18** (Bucket: mlops-295610-mlops-bucket)
- [x] Set up bucket structure (data/, models/, outputs/, pipelines/, etc.) ✅ **Completed 2025-11-18**
- [x] Configure bucket permissions and IAM policies ✅ **Completed 2025-11-18**
- [x] Test file upload/download operations with gsutil ✅ **Completed 2025-11-18**

## Phase 2: Data Pipeline Implementation (Week 2)

### Task 2.1: Dataset Preparation
- [x] Select and download small public dataset (CIFAR-10 subset or similar) ✅ **In Progress 2025-11-18**
- [ ] Upload raw data to GCS bucket (notebook cell ready to execute)
- [x] Create data exploration notebook in Vertex AI Workbench ✅ **Completed 2025-11-18**
- [ ] Document data schema and characteristics

### Task 2.2: Data Processing Pipeline
- [ ] Create custom training job for data preprocessing
- [ ] Implement data validation and quality checks
- [ ] Set up train/validation/test data splits
- [ ] Create preprocessing container and push to Container Registry

### Task 2.3: Data Annotation Workflow (Optional)
- [ ] Research Vertex AI Data Labeling service
- [ ] Create basic data labeling workflow if needed
- [ ] Validate labeled data quality
- [ ] Store processed data in GCS

## Phase 3: Model Training Pipeline (Week 3)

### Task 3.1: Training Script Development
- [ ] Create training script using TensorFlow/PyTorch
- [ ] Implement model evaluation metrics
- [ ] Add hyperparameter configuration
- [ ] Test training script locally

### Task 3.2: Vertex AI Custom Training Job
- [ ] Create Vertex AI Custom Training job configuration
- [ ] Set up pre-built container (TensorFlow/PyTorch)
- [ ] Configure hyperparameters and machine types
- [ ] Execute first training job and verify outputs

### Task 3.3: Model Evaluation and Registry
- [ ] Implement model evaluation with Vertex AI TensorBoard
- [ ] Generate performance metrics and visualizations
- [ ] Register model in Vertex AI Model Registry
- [ ] Create model evaluation report

## Phase 4: Pipeline Orchestration (Week 4)

### Task 4.1: Vertex AI Pipelines Setup
- [ ] Study Kubeflow Pipelines (KFP) and Vertex AI Pipelines docs
- [ ] Install KFP SDK and create first simple pipeline
- [ ] Define pipeline components for data processing → training
- [ ] Test pipeline execution in Vertex AI

### Task 4.2: Model Registry and Versioning
- [ ] Integrate model registration into pipeline
- [ ] Set up model versioning and metadata tracking
- [ ] Implement model approval workflow
- [ ] Test model version management

### Task 4.3: Deployment Pipeline
- [ ] Create model deployment component
- [ ] Set up Vertex AI Endpoint configuration
- [ ] Deploy model to managed endpoint
- [ ] Test inference with sample requests

## Phase 5: Monitoring & Operations (Week 5)

### Task 5.1: Model Monitoring
- [ ] Set up Vertex AI Model Monitoring for drift detection
- [ ] Configure Cloud Monitoring alerts and dashboards
- [ ] Create monitoring for endpoint performance
- [ ] Test drift detection with sample data changes

### Task 5.2: End-to-End Pipeline Integration
- [ ] Combine all components into complete Vertex AI Pipeline
- [ ] Add conditional logic for model approval/deployment
- [ ] Implement automated retraining triggers
- [ ] Test full pipeline execution

### Task 5.3: CI/CD and Documentation
- [ ] Set up Cloud Build for pipeline CI/CD
- [ ] Create comprehensive README with setup instructions
- [ ] Document all pipeline components and configurations
- [ ] Set up cost optimization and resource cleanup

## Quick Start Checklist (MVP - Week 1-2)

### Immediate Actions (Day 1-3)
- [ ] Apply for student credits (GCP Free Tier + Education credits)
- [ ] Clone or create project repository
- [ ] Set up Google Cloud project and enable billing
- [ ] Install required Python packages (google-cloud-aiplatform, kfp, tensorflow, pandas)
- [ ] Create GCS bucket and test connectivity
- [ ] Download sample dataset and upload to GCS

### First Pipeline (Day 4-7)
- [ ] Create simple custom training job (basic classifier)
- [ ] Run first Vertex AI Training job
- [ ] Deploy model to Vertex AI Endpoint
- [ ] Test inference with sample request
- [ ] Document the basic workflow

## Student Access & Cost Optimization

### Free Credits & Programs
1. **Google Cloud Free Tier**
   - $300 credit for new accounts (90-day limit)
   - Always Free tier with ongoing monthly limits
   - Sign up at: https://cloud.google.com/free

2. **GitHub Student Developer Pack**
   - Additional GCP credits for students
   - Requires .edu email or student verification
   - Apply at: https://education.github.com/pack

3. **Google Cloud for Education**
   - Institutional program for schools/universities
   - Check with your school's IT department
   - May provide classroom credits and extended access

4. **Coursera/edX Course Credits**
   - Some online courses include temporary GCP access
   - Look for Google Cloud-sponsored ML/AI courses

### Cost Management Tips
- Set up billing alerts for $5, $25, $50 thresholds
- Use smallest machine types for development (e2-micro, e2-small)
- Delete resources immediately after testing
- Use preemptible instances when possible
- Store data in Coldline/Archive storage classes when not actively used

## Dependencies & Prerequisites
- Google Cloud Platform account with billing enabled
- **Student Access**: Apply for GCP Free Tier ($300 credit) + GitHub Student Developer Pack
- **Educational Credits**: Check if your institution has Google Cloud for Education program
- Python 3.8+ development environment
- Basic understanding of machine learning concepts
- Familiarity with Python, pandas, TensorFlow/PyTorch
- Google Cloud CLI installed and configured
- Jupyter Notebook environment

## Resource Management
- **Machine Types**: Use smallest instances (n1-standard-4, e2-standard-4) for development
- **Endpoints**: Use single node deployments
- **Storage**: Minimize GCS storage and clean up regularly
- **Monitoring**: Basic Cloud Monitoring, avoid premium features initially
- **Scheduling**: Delete endpoints when not in use to minimize costs

## Success Metrics
- [ ] Complete pipeline executes without errors
- [ ] Model successfully deploys to Vertex AI Endpoint
- [ ] Inference endpoint responds correctly to test requests
- [ ] All GCP resources properly configured and accessible
- [ ] Documentation allows reproduction of entire workflow

## Discovered During Work

### Additional Tasks & Learnings (Added during development)
- [x] **Switched from enmacc work email to personal Gmail** ✅ **Completed 2025-11-18**
  - Created new project: mlops-295610
  - Configured authentication for farishussain049@gmail.com
  - Set up billing account linking
- [x] **Comprehensive environment verification notebook created** ✅ **Completed 2025-11-18**
  - Built 01_getting_started.ipynb with 10 sections
  - Includes authentication, API enablement, storage setup, and connectivity tests
  - Dataset preparation with CIFAR-10 subset for learning
- [x] **Environment troubleshooting and fixes** ✅ **Completed 2025-11-18**
  - Fixed syntax errors in notebook cells
  - Resolved billing account linking issues
  - Verified all GCP service connectivity

### Notes & Recommendations
- Project setup took approximately 2-3 hours due to authentication switching
- Personal Gmail account provides better isolation for learning project
- Comprehensive verification notebook saves significant time for future setup
- Free tier ($300 credit) is sufficient for entire learning project if managed properly
