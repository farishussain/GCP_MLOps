#!/bin/bash

# Google Cloud MLOps Project Setup Script
# This script sets up the complete Google Cloud Platform infrastructure for the MLOps pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

# Configuration variables (customize these)
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"  # Set your project ID
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
BUCKET_NAME="${GCS_BUCKET:-mlops-vertex-ai-bucket-$(date +%s)}"
SERVICE_ACCOUNT_NAME="mlops-service-account"
ARTIFACT_REPO_NAME="mlops-training"

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    print_error "Please set your Google Cloud Project ID!"
    echo "You can do this by:"
    echo "  export GCP_PROJECT_ID='your-actual-project-id'"
    echo "  Or edit this script and replace 'your-project-id' with your project ID"
    exit 1
fi

print_header "Google Cloud MLOps Pipeline Setup"

print_status "Project ID: $PROJECT_ID"
print_status "Region: $REGION"
print_status "Zone: $ZONE"
print_status "Bucket Name: $BUCKET_NAME"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "Google Cloud CLI (gcloud) is not installed!"
    print_status "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

print_header "Step 1: Authentication and Project Setup"

# Authenticate with Google Cloud
print_status "Authenticating with Google Cloud..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_status "No active authentication found. Please login..."
    gcloud auth login
fi

# Set the project
print_status "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Verify project exists
if ! gcloud projects describe $PROJECT_ID &> /dev/null; then
    print_error "Project $PROJECT_ID does not exist or you don't have access!"
    exit 1
fi

print_success "Project setup complete"

print_header "Step 2: Enable Required APIs"

apis=(
    "aiplatform.googleapis.com"
    "storage.googleapis.com"
    "cloudbuild.googleapis.com"
    "containerregistry.googleapis.com"
    "artifactregistry.googleapis.com"
    "compute.googleapis.com"
    "iam.googleapis.com"
    "logging.googleapis.com"
    "monitoring.googleapis.com"
)

for api in "${apis[@]}"; do
    print_status "Enabling $api..."
    gcloud services enable $api
done

print_success "All required APIs enabled"

print_header "Step 3: Create Service Account"

# Check if service account exists
if gcloud iam service-accounts describe ${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com &> /dev/null; then
    print_warning "Service account $SERVICE_ACCOUNT_NAME already exists"
else
    print_status "Creating service account: $SERVICE_ACCOUNT_NAME..."
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="MLOps Pipeline Service Account" \
        --description="Service account for MLOps pipeline operations"
fi

# Assign IAM roles to service account
roles=(
    "roles/aiplatform.user"
    "roles/storage.admin"
    "roles/artifactregistry.admin"
    "roles/cloudbuild.builds.editor"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
    "roles/ml.admin"
)

for role in "${roles[@]}"; do
    print_status "Assigning role: $role..."
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="$role"
done

print_success "Service account created and configured"

print_header "Step 4: Create Cloud Storage Bucket"

# Check if bucket exists
if gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    print_warning "Bucket gs://$BUCKET_NAME already exists"
else
    print_status "Creating Cloud Storage bucket: $BUCKET_NAME..."
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME
    
    # Create directory structure
    print_status "Creating bucket directory structure..."
    echo "placeholder" | gsutil cp - gs://$BUCKET_NAME/data/placeholder.txt
    echo "placeholder" | gsutil cp - gs://$BUCKET_NAME/models/placeholder.txt
    echo "placeholder" | gsutil cp - gs://$BUCKET_NAME/pipelines/placeholder.txt
    echo "placeholder" | gsutil cp - gs://$BUCKET_NAME/outputs/placeholder.txt
fi

print_success "Cloud Storage bucket configured"

print_header "Step 5: Create Artifact Registry Repository"

# Check if repository exists
if gcloud artifacts repositories describe $ARTIFACT_REPO_NAME --location=$REGION &> /dev/null; then
    print_warning "Artifact Registry repository $ARTIFACT_REPO_NAME already exists"
else
    print_status "Creating Artifact Registry repository: $ARTIFACT_REPO_NAME..."
    gcloud artifacts repositories create $ARTIFACT_REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Repository for MLOps training containers"
fi

print_success "Artifact Registry repository created"

print_header "Step 6: Configure Authentication for Docker"

print_status "Configuring Docker authentication for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

print_success "Docker authentication configured"

print_header "Step 7: Download Service Account Key"

KEY_FILE="service-account-key.json"
if [ ! -f "$KEY_FILE" ]; then
    print_status "Downloading service account key..."
    gcloud iam service-accounts keys create $KEY_FILE \
        --iam-account=${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com
    
    print_warning "Service account key saved as: $KEY_FILE"
    print_warning "Keep this file secure and do not commit to version control!"
else
    print_warning "Service account key file already exists: $KEY_FILE"
fi

print_header "Step 8: Update Configuration Files"

# Update config.yaml
print_status "Updating configs/config.yaml..."
cat > configs/config.yaml << EOF
# Google Cloud MLOps Configuration

# Project Settings
project:
  name: "mlops-demo"
  description: "End-to-end MLOps pipeline with Google Vertex AI"
  version: "1.0.0"

# Google Cloud Configuration  
gcp:
  project_id: "$PROJECT_ID"
  region: "$REGION"
  zone: "$ZONE"
  service_account: "${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
  
# Cloud Storage Configuration
storage:
  bucket_name: "$BUCKET_NAME"
  data_path: "data/"
  models_path: "models/"
  pipelines_path: "pipelines/"
  outputs_path: "outputs/"

# Vertex AI Configuration
vertex_ai:
  location: "$REGION"
  staging_bucket: "$BUCKET_NAME"
  
# Model Configuration
model:
  name: "iris-classifier"
  framework: "scikit-learn"
  version: "v1.0"
  
# Training Configuration  
training:
  dataset: "iris"
  test_size: 0.2
  random_state: 42
  
# Pipeline Configuration
pipeline:
  name: "iris-mlops-pipeline"
  description: "End-to-end Iris classification pipeline"
  
# Container Configuration
container:
  image_uri: "${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO_NAME}/iris-trainer"
  tag: "latest"
EOF

# Create .env file for environment variables
print_status "Creating .env file..."
cat > .env << EOF
# Google Cloud Platform Configuration
GCP_PROJECT_ID=$PROJECT_ID
GCP_REGION=$REGION
GCP_ZONE=$ZONE
GCS_BUCKET=$BUCKET_NAME
GOOGLE_APPLICATION_CREDENTIALS=service-account-key.json

# Vertex AI Configuration
VERTEX_AI_LOCATION=$REGION
VERTEX_AI_STAGING_BUCKET=$BUCKET_NAME

# Container Configuration
ARTIFACT_REGISTRY_REGION=$REGION
ARTIFACT_REPO_NAME=$ARTIFACT_REPO_NAME
EOF

print_success "Configuration files updated"

print_header "Step 9: Set Up Python Environment"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
fi

print_status "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

print_success "Python environment configured"

print_header "Step 10: Verify Setup"

# Set environment variables for verification
export GOOGLE_APPLICATION_CREDENTIALS="service-account-key.json"
export GCP_PROJECT_ID=$PROJECT_ID
export GCS_BUCKET=$BUCKET_NAME

print_status "Running setup verification..."

# Test Google Cloud authentication
print_status "Testing Google Cloud authentication..."
gcloud auth list

# Test Vertex AI access
print_status "Testing Vertex AI access..."
gcloud ai models list --region=$REGION

# Test Cloud Storage access
print_status "Testing Cloud Storage access..."
gsutil ls gs://$BUCKET_NAME/

print_success "Setup verification complete"

print_header "Setup Complete!"

print_success "🎉 Google Cloud MLOps pipeline setup completed successfully!"
echo ""
print_status "Next steps:"
echo "1. Activate the Python environment: source venv/bin/activate"
echo "2. Set environment variables: source .env"
echo "3. Run the notebooks in order:"
echo "   - notebooks/01_getting_started.ipynb"
echo "   - notebooks/02_data_processing_pipeline.ipynb"
echo "   - notebooks/03_model_training.ipynb"
echo "   - notebooks/04_vertex_ai_training.ipynb"
echo "   - notebooks/05_model_deployment.ipynb"
echo "   - notebooks/06_vertex_ai_pipelines.ipynb"
echo ""
print_status "Configuration Details:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Bucket: gs://$BUCKET_NAME"
echo "  Service Account: ${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
echo "  Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO_NAME}"
echo ""
print_warning "Important Security Notes:"
echo "- Keep service-account-key.json secure and never commit it to version control"
echo "- Add service-account-key.json to your .gitignore file"
echo "- Consider using Workload Identity Federation for production environments"
echo ""
print_status "For troubleshooting and documentation, visit:"
echo "- https://cloud.google.com/vertex-ai/docs"
echo "- https://cloud.google.com/storage/docs"
echo "- https://cloud.google.com/artifact-registry/docs"
