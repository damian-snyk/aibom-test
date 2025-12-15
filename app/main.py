"""
AIBOM Test Application - FastAPI Entry Point

A sample AI/ML application demonstrating AWS Bedrock and AgentCore integration.
"""
import os
from contextlib import asynccontextmanager
from typing import Any

import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BEDROCK_AGENT_ID = os.getenv("BEDROCK_AGENT_ID", "")
BEDROCK_AGENT_ALIAS_ID = os.getenv("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print(f"Starting AIBOM Application ({ENVIRONMENT})")
    yield
    # Shutdown
    print("Shutting down AIBOM Application")


app = FastAPI(
    title="AIBOM Test Application",
    description="AI Bill of Materials Test Application with AWS Bedrock",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


class InvokeRequest(BaseModel):
    prompt: str
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    max_tokens: int = 4096


class InvokeResponse(BaseModel):
    response: str
    model: str
    usage: dict[str, Any] | None = None


class AgentRequest(BaseModel):
    input: str
    session_id: str | None = None


class AgentResponse(BaseModel):
    response: str
    session_id: str


# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        environment=ENVIRONMENT,
        version="1.0.0"
    )


# Bedrock model invocation
@app.post("/invoke", response_model=InvokeResponse)
async def invoke_model(request: InvokeRequest):
    """Invoke a Bedrock model directly."""
    try:
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        
        import json
        response = bedrock_runtime.invoke_model(
            modelId=request.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": request.max_tokens,
                "messages": [
                    {"role": "user", "content": request.prompt}
                ]
            }),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response["body"].read())
        
        return InvokeResponse(
            response=response_body.get("content", [{}])[0].get("text", ""),
            model=request.model_id,
            usage=response_body.get("usage")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Bedrock Agent invocation
@app.post("/agent", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """Invoke the Bedrock Agent."""
    if not BEDROCK_AGENT_ID:
        raise HTTPException(status_code=503, detail="Bedrock Agent not configured")
    
    try:
        import uuid
        bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
        
        session_id = request.session_id or str(uuid.uuid4())
        
        response = bedrock_agent_runtime.invoke_agent(
            agentId=BEDROCK_AGENT_ID,
            agentAliasId=BEDROCK_AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=request.input,
        )
        
        # Collect response chunks
        output_text = ""
        for event in response.get("completion", []):
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    output_text += chunk["bytes"].decode("utf-8")
        
        return AgentResponse(
            response=output_text,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# List available models
@app.get("/models")
async def list_models():
    """List available Bedrock foundation models."""
    try:
        bedrock = boto3.client("bedrock", region_name=AWS_REGION)
        
        response = bedrock.list_foundation_models(
            byOutputModality="TEXT"
        )
        
        models = [
            {
                "model_id": model["modelId"],
                "model_name": model["modelName"],
                "provider": model["providerName"],
                "input_modalities": model.get("inputModalities", []),
                "output_modalities": model.get("outputModalities", []),
            }
            for model in response.get("modelSummaries", [])
        ]
        
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

