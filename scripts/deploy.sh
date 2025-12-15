#!/bin/bash
# =============================================================================
# AWS Deployment Script
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${PROJECT_NAME:-aibom-app}"

# Functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    deploy-infra    Deploy infrastructure with CDK
    deploy-app      Deploy application to ECS
    deploy-agent    Deploy Bedrock Agent
    deploy-all      Deploy everything
    destroy         Destroy all infrastructure

Options:
    -e, --environment   Environment (staging|production) [default: staging]
    -r, --region        AWS region [default: us-east-1]
    -h, --help          Show this help message

Examples:
    $0 deploy-all -e production
    $0 deploy-infra -e staging
    $0 deploy-app -e staging
EOF
}

check_aws_credentials() {
    log_info "Checking AWS credentials..."
    
    if ! aws sts get-caller-identity &>/dev/null; then
        log_error "AWS credentials not configured or expired"
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    log_success "Using AWS Account: $ACCOUNT_ID"
}

check_requirements() {
    log_info "Checking requirements..."
    
    local missing=()
    command -v aws >/dev/null 2>&1 || missing+=("aws-cli")
    command -v cdk >/dev/null 2>&1 || missing+=("aws-cdk")
    command -v docker >/dev/null 2>&1 || missing+=("docker")
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing tools: ${missing[*]}"
        exit 1
    fi
    
    log_success "All requirements met"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure to $ENVIRONMENT..."
    
    cd "$PROJECT_ROOT/infrastructure"
    
    # Install CDK dependencies
    pip install -r requirements.txt -q
    
    # Bootstrap CDK (if needed)
    log_info "Bootstrapping CDK..."
    cdk bootstrap aws://$ACCOUNT_ID/$AWS_REGION --context environment=$ENVIRONMENT || true
    
    # Synthesize
    log_info "Synthesizing CloudFormation templates..."
    cdk synth --context environment=$ENVIRONMENT
    
    # Show diff
    log_info "Showing infrastructure changes..."
    cdk diff --context environment=$ENVIRONMENT || true
    
    # Deploy
    log_info "Deploying infrastructure..."
    cdk deploy --all \
        --require-approval never \
        --context environment=$ENVIRONMENT \
        --outputs-file "$PROJECT_ROOT/cdk-outputs.json"
    
    log_success "Infrastructure deployed successfully"
    
    cd "$PROJECT_ROOT"
}

build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    ECR_REPO="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME"
    IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | \
        docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names $PROJECT_NAME --region $AWS_REGION &>/dev/null || \
        aws ecr create-repository --repository-name $PROJECT_NAME --region $AWS_REGION
    
    # Build image
    log_info "Building Docker image..."
    docker build -t $PROJECT_NAME:$IMAGE_TAG "$PROJECT_ROOT"
    
    # Tag and push
    docker tag $PROJECT_NAME:$IMAGE_TAG $ECR_REPO:$IMAGE_TAG
    docker tag $PROJECT_NAME:$IMAGE_TAG $ECR_REPO:latest
    
    log_info "Pushing to ECR..."
    docker push $ECR_REPO:$IMAGE_TAG
    docker push $ECR_REPO:latest
    
    log_success "Image pushed: $ECR_REPO:$IMAGE_TAG"
}

deploy_application() {
    log_info "Deploying application to ECS..."
    
    build_and_push_image
    
    CLUSTER_NAME="aibom-cluster-$ENVIRONMENT"
    SERVICE_NAME="aibom-service-$ENVIRONMENT"
    
    # Update ECS service
    log_info "Updating ECS service..."
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --force-new-deployment \
        --region $AWS_REGION
    
    # Wait for deployment
    log_info "Waiting for deployment to stabilize..."
    aws ecs wait services-stable \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME \
        --region $AWS_REGION
    
    log_success "Application deployed successfully"
}

deploy_bedrock_agent() {
    log_info "Deploying Bedrock Agent..."
    
    CONFIG_FILE="$PROJECT_ROOT/config/bedrock-agent-$ENVIRONMENT.yaml"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    python3 "$SCRIPT_DIR/deploy_bedrock_agent.py" \
        --config "$CONFIG_FILE" \
        --environment $ENVIRONMENT \
        --region $AWS_REGION
    
    log_success "Bedrock Agent deployed successfully"
}

deploy_all() {
    log_info "Starting full deployment to $ENVIRONMENT..."
    
    deploy_infrastructure
    deploy_application
    deploy_bedrock_agent
    
    log_success "Full deployment completed!"
    
    # Print outputs
    if [ -f "$PROJECT_ROOT/cdk-outputs.json" ]; then
        log_info "Deployment outputs:"
        cat "$PROJECT_ROOT/cdk-outputs.json"
    fi
}

destroy_infrastructure() {
    log_warning "This will destroy all infrastructure in $ENVIRONMENT"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Cancelled"
        exit 0
    fi
    
    cd "$PROJECT_ROOT/infrastructure"
    
    cdk destroy --all \
        --force \
        --context environment=$ENVIRONMENT
    
    log_success "Infrastructure destroyed"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        deploy-infra|deploy-app|deploy-agent|deploy-all|destroy)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT (must be staging or production)"
    exit 1
fi

# Run command
check_requirements
check_aws_credentials

case $COMMAND in
    deploy-infra)
        deploy_infrastructure
        ;;
    deploy-app)
        deploy_application
        ;;
    deploy-agent)
        deploy_bedrock_agent
        ;;
    deploy-all)
        deploy_all
        ;;
    destroy)
        destroy_infrastructure
        ;;
    *)
        usage
        exit 1
        ;;
esac

