#!/bin/bash
# =============================================================================
# Local Development Setup Script
# =============================================================================
set -e

echo "=========================================="
echo "AIBOM Test Application - Local Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
check_requirements() {
    echo -e "\n${YELLOW}Checking requirements...${NC}"
    
    local missing=()
    
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    command -v pip >/dev/null 2>&1 || missing+=("pip")
    command -v docker >/dev/null 2>&1 || missing+=("docker")
    command -v docker-compose >/dev/null 2>&1 || missing+=("docker-compose")
    command -v aws >/dev/null 2>&1 || missing+=("aws-cli")
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo -e "${RED}Missing required tools: ${missing[*]}${NC}"
        echo "Please install the missing tools and try again."
        exit 1
    fi
    
    echo -e "${GREEN}All requirements met!${NC}"
}

# Create virtual environment
setup_venv() {
    echo -e "\n${YELLOW}Setting up Python virtual environment...${NC}"
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}Virtual environment created!${NC}"
    else
        echo "Virtual environment already exists."
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo -e "${GREEN}Dependencies installed!${NC}"
}

# Setup environment variables
setup_env() {
    echo -e "\n${YELLOW}Setting up environment variables...${NC}"
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo -e "${GREEN}Created .env from .env.example${NC}"
            echo -e "${YELLOW}Please update .env with your actual values${NC}"
        else
            echo -e "${RED}.env.example not found, creating basic .env${NC}"
            cat > .env << 'EOF'
# AWS Configuration
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# API Keys (optional for local testing)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Local Services
REDIS_URL=redis://localhost:6379
CHROMA_URL=http://localhost:8000
EOF
        fi
    else
        echo ".env file already exists."
    fi
}

# Start local services
start_services() {
    echo -e "\n${YELLOW}Starting local services with Docker Compose...${NC}"
    
    docker-compose up -d localstack chromadb redis
    
    echo -e "${GREEN}Local services started!${NC}"
    echo "LocalStack: http://localhost:4566"
    echo "ChromaDB:   http://localhost:8000"
    echo "Redis:      localhost:6379"
}

# Setup LocalStack resources
setup_localstack() {
    echo -e "\n${YELLOW}Setting up LocalStack resources...${NC}"
    
    # Wait for LocalStack to be ready
    echo "Waiting for LocalStack..."
    until curl -s http://localhost:4566/_localstack/health | grep -q "running"; do
        sleep 2
    done
    
    # Create S3 buckets
    aws --endpoint-url=http://localhost:4566 s3 mb s3://aibom-app-development-data 2>/dev/null || true
    aws --endpoint-url=http://localhost:4566 s3 mb s3://aibom-app-development-knowledge-base 2>/dev/null || true
    
    # Create DynamoDB tables
    aws --endpoint-url=http://localhost:4566 dynamodb create-table \
        --table-name aibom-app-development-conversations \
        --attribute-definitions \
            AttributeName=conversation_id,AttributeType=S \
            AttributeName=timestamp,AttributeType=S \
        --key-schema \
            AttributeName=conversation_id,KeyType=HASH \
            AttributeName=timestamp,KeyType=RANGE \
        --billing-mode PAY_PER_REQUEST 2>/dev/null || true
    
    echo -e "${GREEN}LocalStack resources created!${NC}"
}

# Main
main() {
    cd "$(dirname "$0")/.."
    
    check_requirements
    setup_venv
    setup_env
    start_services
    setup_localstack
    
    echo -e "\n${GREEN}=========================================="
    echo "Local setup complete!"
    echo "==========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Update .env with your AWS credentials and API keys"
    echo "2. Activate the virtual environment: source venv/bin/activate"
    echo "3. Run the application: python -m uvicorn app.main:app --reload"
    echo ""
    echo "To stop services: docker-compose down"
}

main "$@"

