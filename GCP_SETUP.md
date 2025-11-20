# Google Cloud Platform Setup Guide

This guide will help you set up the complete Google Cloud MLOps pipeline infrastructure.

## Prerequisites

Before starting, ensure you have:

1. **Google Cloud Project**: A GCP project with billing enabled
2. **Google Cloud CLI**: Install from [here](https://cloud.google.com/sdk/docs/install)
3. **Python 3.8+**: For running the MLOps pipeline
4. **Docker** (optional): For custom training containers

## Quick Setup

### 1. Set Your Project ID

First, set your Google Cloud Project ID as an environment variable:

```bash
export GCP_PROJECT_ID="your-actual-project-id"
```

Replace `your-actual-project-id` with your real GCP project ID.

### 2. Run the Setup Script

Execute the automated setup script:

```bash
./setup_gcp.sh
```

This script will:
- ✅ Enable all required Google Cloud APIs
- ✅ Create a service account with appropriate permissions
- ✅ Set up Cloud Storage bucket with proper structure
- ✅ Create Artifact Registry repository for containers
- ✅ Configure authentication
- ✅ Update configuration files
- ✅ Set up Python environment
- ✅ Verify the complete setup

## Manual Setup (Alternative)

If you prefer to set up manually, follow these steps:

### 1. Authentication

```bash
# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable APIs

```bash
# Enable required Google Cloud APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
```

### 3. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create mlops-service-account \
    --display-name="MLOps Pipeline Service Account" \
    --description="Service account for MLOps pipeline operations"

# Assign roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:mlops-service-account@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:mlops-service-account@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:mlops-service-account@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/ml.admin"
```

### 4. Create Cloud Storage Bucket

```bash
# Create bucket (use a unique name)
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://your-unique-bucket-name

# Create directory structure
echo "placeholder" | gsutil cp - gs://your-unique-bucket-name/data/placeholder.txt
echo "placeholder" | gsutil cp - gs://your-unique-bucket-name/models/placeholder.txt
echo "placeholder" | gsutil cp - gs://your-unique-bucket-name/pipelines/placeholder.txt
echo "placeholder" | gsutil cp - gs://your-unique-bucket-name/outputs/placeholder.txt
```

### 5. Create Artifact Registry Repository

```bash
# Create repository for Docker containers
gcloud artifacts repositories create mlops-training \
    --repository-format=docker \
    --location=us-central1 \
    --description="Repository for MLOps training containers"

# Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 6. Download Service Account Key

```bash
# Download service account key
gcloud iam service-accounts keys create service-account-key.json \
    --iam-account=mlops-service-account@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

⚠️ **Important**: Keep this key file secure and never commit it to version control!

### 7. Update Configuration

Update `configs/config.yaml` with your project details:

```yaml
gcp:
  project_id: "YOUR_PROJECT_ID"
  region: "us-central1"
  zone: "us-central1-a"

storage:
  bucket_name: "your-unique-bucket-name"

vertex_ai:
  location: "us-central1"
  staging_bucket: "your-unique-bucket-name"
```

### 8. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 9. Set Environment Variables

Create a `.env` file or export variables:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="service-account-key.json"
export GCP_PROJECT_ID="YOUR_PROJECT_ID"
export GCS_BUCKET="your-unique-bucket-name"
```

## Verification

After setup, verify everything works:

```bash
# Check authentication
gcloud auth list

# Test Vertex AI access
gcloud ai models list --region=us-central1

# Test storage access
gsutil ls gs://your-unique-bucket-name/

# Test Python environment
python -c "from google.cloud import aiplatform; print('Vertex AI SDK imported successfully')"
```

## Running the Pipeline

After successful setup, run the notebooks in order:

1. `notebooks/01_getting_started.ipynb` - Environment verification
2. `notebooks/02_data_processing_pipeline.ipynb` - Data processing
3. `notebooks/03_model_training.ipynb` - Model training
4. `notebooks/04_vertex_ai_training.ipynb` - Cloud training
5. `notebooks/05_model_deployment.ipynb` - Model deployment
6. `notebooks/06_vertex_ai_pipelines.ipynb` - Pipeline orchestration

## Cost Management

To minimize costs during development:

### Free Tier Resources
- Cloud Storage: 5 GB free per month
- Vertex AI: Limited free usage
- Compute Engine: 1 f1-micro instance free

### Cost Optimization Tips
1. **Use preemptible instances** for training jobs
2. **Set up budget alerts** in Google Cloud Console
3. **Clean up resources** regularly:
   ```bash
   # Delete unused models
   gcloud ai models list --region=us-central1
   gcloud ai models delete MODEL_ID --region=us-central1
   
   # Delete unused endpoints
   gcloud ai endpoints list --region=us-central1
   gcloud ai endpoints delete ENDPOINT_ID --region=us-central1
   ```
4. **Monitor usage** in the GCP Console

### Budget Setup
```bash
# Create budget alert (replace with your billing account)
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="MLOps Pipeline Budget" \
    --budget-amount=50 \
    --threshold-rule=percent=80 \
    --threshold-rule=percent=100
```

## Troubleshooting

### Common Issues

1. **Permission denied errors**
   - Ensure your service account has the correct roles
   - Verify your authentication: `gcloud auth list`

2. **Quota exceeded errors**
   - Check your project quotas in GCP Console
   - Request quota increases if needed

3. **API not enabled errors**
   - Run the setup script again
   - Manually enable APIs from GCP Console

4. **Storage access errors**
   - Verify bucket exists: `gsutil ls gs://your-bucket-name/`
   - Check service account permissions

5. **Container registry issues**
   - Configure Docker auth: `gcloud auth configure-docker us-central1-docker.pkg.dev`
   - Verify repository exists: `gcloud artifacts repositories list`

### Getting Help

- **Google Cloud Documentation**: https://cloud.google.com/docs
- **Vertex AI Documentation**: https://cloud.google.com/vertex-ai/docs
- **MLOps Best Practices**: https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

## Security Best Practices

1. **Service Account Keys**
   - Store keys securely
   - Use environment variables, not hardcoded paths
   - Consider Workload Identity Federation for production

2. **IAM Permissions**
   - Follow principle of least privilege
   - Regularly audit permissions
   - Use IAM conditions for fine-grained access

3. **Data Protection**
   - Enable bucket versioning
   - Set up retention policies
   - Use encryption at rest and in transit

4. **Network Security**
   - Use VPC for production deployments
   - Set up firewalls and network policies
   - Consider Private Google Access

## Next Steps

After successful setup:

1. Explore the Jupyter notebooks
2. Customize the pipeline for your data
3. Implement custom training algorithms
4. Set up monitoring and alerting
5. Create CI/CD pipelines for automated deployment

## Resources

- **Project Documentation**: See `PLANNING.md` for architecture details
- **API Reference**: https://cloud.google.com/vertex-ai/docs/reference
- **Sample Code**: https://github.com/GoogleCloudPlatform/vertex-ai-samples
- **Best Practices**: https://cloud.google.com/vertex-ai/docs/best-practices
