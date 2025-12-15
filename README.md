# AIBOM Test Application

A comprehensive AI/ML application demonstrating integration with AWS Bedrock, AgentCore, and various AI frameworks for testing AI Bill of Materials (AIBOM) detection.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │   Lint   │→ │   Test   │→ │ Security │→ │  Deploy  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                           VPC                                │   │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────────────┐   │   │
│  │  │    ALB    │───▶│    ECS    │───▶│  Bedrock/AgentCore│   │   │
│  │  └───────────┘    │  Fargate  │    └───────────────────┘   │   │
│  │                   └───────────┘              │              │   │
│  │                         │                    ▼              │   │
│  │  ┌───────────┐    ┌─────┴─────┐    ┌───────────────────┐   │   │
│  │  │  Lambda   │    │    S3     │    │  Knowledge Base   │   │   │
│  │  └───────────┘    │ DynamoDB  │    │  (OpenSearch)     │   │   │
│  │        │          │   Redis   │    └───────────────────┘   │   │
│  │        ▼          └───────────┘                            │   │
│  │  ┌───────────┐                                             │   │
│  │  │    API    │                                             │   │
│  │  │  Gateway  │                                             │   │
│  │  └───────────┘                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
aibom-test/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI pipeline (lint, test, security)
│       ├── cd-deploy.yml          # CD pipeline (deploy to AWS)
│       └── security-scan.yml      # Scheduled security scans
├── app/
│   ├── __init__.py
│   └── main.py                    # FastAPI application
├── config/
│   ├── bedrock-agent-staging.yaml
│   └── bedrock-agent-production.yaml
├── infrastructure/
│   ├── app.py                     # CDK application entry
│   ├── cdk.json                   # CDK configuration
│   ├── requirements.txt           # CDK dependencies
│   └── stacks/
│       ├── network_stack.py       # VPC, subnets, security groups
│       ├── security_stack.py      # IAM roles, secrets
│       ├── storage_stack.py       # S3, DynamoDB, ElastiCache
│       ├── compute_stack.py       # ECS, Lambda, API Gateway
│       ├── bedrock_stack.py       # Bedrock agents, guardrails
│       └── monitoring_stack.py    # CloudWatch, alarms
├── scripts/
│   ├── deploy.sh                  # Main deployment script
│   ├── deploy_bedrock_agent.py    # Bedrock agent deployment
│   ├── setup-local.sh             # Local development setup
│   └── run-security-scans.sh      # Security scanning
├── tests/
│   └── integration/               # Integration tests
├── Dockerfile                     # Multi-stage Docker build
├── docker-compose.yml             # Local development services
├── requirements.txt               # Python dependencies
├── sample_bedrock_app.py          # Sample AI/ML code
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- AWS CLI configured
- Node.js 18+ (for CDK)
- Snyk CLI (for security scans)

### Local Development

```bash
# 1. Clone and setup
git clone <repository-url>
cd aibom-test

# 2. Run setup script
./scripts/setup-local.sh

# 3. Update environment variables
cp env.example .env
# Edit .env with your credentials

# 4. Start the application
source venv/bin/activate
python -m uvicorn app.main:app --reload
```

### Deploy to AWS

```bash
# Deploy infrastructure
./scripts/deploy.sh deploy-infra -e staging

# Deploy application
./scripts/deploy.sh deploy-app -e staging

# Deploy Bedrock agent
./scripts/deploy.sh deploy-agent -e staging

# Or deploy everything at once
./scripts/deploy.sh deploy-all -e staging
```

## 🔐 Security

### Snyk Integration

The project includes comprehensive Snyk security scanning:

- **SCA**: Open-source dependency vulnerabilities
- **SAST**: Code security analysis
- **IaC**: Infrastructure as Code security
- **Container**: Docker image vulnerabilities
- **AIBOM**: AI Bill of Materials

Run scans locally:

```bash
./scripts/run-security-scans.sh
```

### Required Secrets (GitHub Actions)

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_DEPLOY_ROLE_ARN` | IAM role for deployments |
| `SNYK_TOKEN` | Snyk API token |
| `SLACK_WEBHOOK_URL` | Slack notifications (optional) |

## 🤖 AI/ML Components

This application demonstrates usage of:

### Libraries
- LangChain & LangChain-AWS
- PyTorch & TorchVision
- TensorFlow & Keras
- Hugging Face Transformers
- LlamaIndex
- spaCy
- OpenAI SDK
- Anthropic SDK

### Models
- AWS Bedrock: Claude, Titan, Llama, Mistral, Cohere
- OpenAI: GPT-4o, DALL-E, Whisper, Embeddings
- Anthropic: Claude 3.5 Sonnet, Claude 3 Opus
- Hugging Face: BERT, GPT-2, Llama, Mistral, etc.

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/invoke` | POST | Invoke Bedrock model |
| `/agent` | POST | Invoke Bedrock Agent |
| `/models` | GET | List available models |

## 🔄 CI/CD Pipeline

### CI Pipeline (on PR/push)
1. **Lint**: Ruff, Black, isort, mypy
2. **Test**: pytest with coverage
3. **Security**: Snyk SCA, SAST
4. **Build**: Docker image build & push to ECR

### CD Pipeline (on merge to main)
1. **Deploy Infrastructure**: CDK deploy
2. **Deploy Application**: ECS service update
3. **Deploy Bedrock Agent**: Agent configuration
4. **Integration Tests**: API validation
5. **Notify**: Slack notification

## 🏷️ Environment Variables

See `env.example` for all available configuration options.

Key variables:

```bash
# AWS
AWS_DEFAULT_REGION=us-east-1

# Bedrock
BEDROCK_AGENT_ID=your-agent-id
DEFAULT_CHAT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## 📝 License

MIT License - see LICENSE file for details.

