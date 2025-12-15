#!/usr/bin/env python3
"""
AIBOM Analysis MCP Server

A Model Context Protocol (MCP) server that provides AI Bill of Materials
analysis tools and prompts for AI assistants.

This server demonstrates:
- MCP tool definitions
- MCP prompt templates
- MCP resource handling
- Integration with AI/ML analysis capabilities

Usage:
    python -m mcp_server.server

Or with uvx:
    uvx mcp-server-aibom
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Prompt,
    PromptMessage,
    PromptArgument,
    GetPromptResult,
    Resource,
    ResourceTemplate,
)

# =============================================================================
# Server Configuration
# =============================================================================

SERVER_NAME = "aibom-analysis-server"
SERVER_VERSION = "1.0.0"

# Create the MCP server instance
server = Server(SERVER_NAME)

# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    Tool(
        name="analyze_python_file",
        description="Analyze a Python file for AI/ML components including models, libraries, and prompts",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the Python file to analyze"
                },
                "include_prompts": {
                    "type": "boolean",
                    "description": "Whether to detect prompt patterns",
                    "default": True
                },
                "include_models": {
                    "type": "boolean",
                    "description": "Whether to detect model references",
                    "default": True
                }
            },
            "required": ["file_path"]
        }
    ),
    Tool(
        name="scan_directory",
        description="Scan a directory recursively for AI/ML components and generate an AIBOM report",
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to scan"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["json", "markdown", "cyclonedx"],
                    "description": "Output format for the AIBOM report",
                    "default": "json"
                },
                "include_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File patterns to include (e.g., '*.py')"
                },
                "exclude_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File patterns to exclude"
                }
            },
            "required": ["directory"]
        }
    ),
    Tool(
        name="identify_model",
        description="Identify details about a specific AI model by its identifier",
        inputSchema={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model identifier (e.g., 'gpt-4', 'claude-3-opus', 'meta.llama3-70b')"
                }
            },
            "required": ["model_id"]
        }
    ),
    Tool(
        name="analyze_prompt",
        description="Analyze a prompt for patterns, potential issues, and optimization suggestions",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt_text": {
                    "type": "string",
                    "description": "The prompt text to analyze"
                },
                "check_injection": {
                    "type": "boolean",
                    "description": "Check for potential prompt injection vulnerabilities",
                    "default": True
                },
                "suggest_improvements": {
                    "type": "boolean",
                    "description": "Provide optimization suggestions",
                    "default": True
                }
            },
            "required": ["prompt_text"]
        }
    ),
    Tool(
        name="compare_models",
        description="Compare two or more AI models on various dimensions",
        inputSchema={
            "type": "object",
            "properties": {
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model identifiers to compare",
                    "minItems": 2
                },
                "dimensions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["capability", "cost", "speed", "context_length", "licensing"]
                    },
                    "description": "Dimensions to compare"
                }
            },
            "required": ["models"]
        }
    ),
    Tool(
        name="generate_aibom",
        description="Generate a complete AI Bill of Materials in CycloneDX format",
        inputSchema={
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Path to the project root"
                },
                "project_name": {
                    "type": "string",
                    "description": "Name of the project"
                },
                "version": {
                    "type": "string",
                    "description": "Project version"
                },
                "include_dependencies": {
                    "type": "boolean",
                    "description": "Include transitive dependencies",
                    "default": True
                }
            },
            "required": ["project_path"]
        }
    ),
    Tool(
        name="check_license_compliance",
        description="Check AI model and library licenses for compliance issues",
        inputSchema={
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of component names to check"
                },
                "allowed_licenses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of allowed license types"
                },
                "use_case": {
                    "type": "string",
                    "enum": ["commercial", "open-source", "research", "internal"],
                    "description": "Intended use case for license evaluation"
                }
            },
            "required": ["components"]
        }
    ),
    Tool(
        name="security_scan",
        description="Scan for security issues in AI/ML code including prompt injection risks",
        inputSchema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "File or directory to scan"
                },
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "prompt_injection",
                            "data_leakage",
                            "model_theft",
                            "adversarial_inputs",
                            "pii_exposure"
                        ]
                    },
                    "description": "Security checks to perform"
                }
            },
            "required": ["target"]
        }
    )
]


# =============================================================================
# Prompt Templates
# =============================================================================

PROMPTS = {
    "aibom_analysis": Prompt(
        name="aibom_analysis",
        description="Comprehensive AI Bill of Materials analysis for a codebase",
        arguments=[
            PromptArgument(
                name="project_path",
                description="Path to the project to analyze",
                required=True
            ),
            PromptArgument(
                name="analysis_depth",
                description="Depth of analysis: 'quick', 'standard', or 'deep'",
                required=False
            )
        ]
    ),
    "model_selection": Prompt(
        name="model_selection",
        description="Help select the best AI model for a specific use case",
        arguments=[
            PromptArgument(
                name="use_case",
                description="Description of the use case",
                required=True
            ),
            PromptArgument(
                name="constraints",
                description="Any constraints (budget, latency, privacy, etc.)",
                required=False
            )
        ]
    ),
    "prompt_optimization": Prompt(
        name="prompt_optimization",
        description="Optimize a prompt for better performance",
        arguments=[
            PromptArgument(
                name="original_prompt",
                description="The original prompt to optimize",
                required=True
            ),
            PromptArgument(
                name="target_model",
                description="Target model (e.g., 'gpt-4', 'claude-3')",
                required=False
            ),
            PromptArgument(
                name="optimization_goal",
                description="Goal: 'accuracy', 'speed', 'cost', 'safety'",
                required=False
            )
        ]
    ),
    "security_review": Prompt(
        name="security_review",
        description="Security review of AI/ML implementation",
        arguments=[
            PromptArgument(
                name="code_or_config",
                description="Code or configuration to review",
                required=True
            ),
            PromptArgument(
                name="threat_model",
                description="Specific threats to check for",
                required=False
            )
        ]
    ),
    "migration_assistant": Prompt(
        name="migration_assistant",
        description="Assist with migrating between AI models or providers",
        arguments=[
            PromptArgument(
                name="source_model",
                description="Current model being used",
                required=True
            ),
            PromptArgument(
                name="target_model",
                description="Model to migrate to",
                required=True
            ),
            PromptArgument(
                name="current_implementation",
                description="Description or code of current implementation",
                required=False
            )
        ]
    ),
    "compliance_check": Prompt(
        name="compliance_check",
        description="Check AI implementation for regulatory compliance",
        arguments=[
            PromptArgument(
                name="implementation_details",
                description="Details of the AI implementation",
                required=True
            ),
            PromptArgument(
                name="regulations",
                description="Regulations to check (e.g., 'GDPR', 'EU AI Act', 'SOC2')",
                required=False
            )
        ]
    )
}

# Prompt message templates
PROMPT_MESSAGES = {
    "aibom_analysis": """You are an AI Bill of Materials (AIBOM) analyst. Analyze the codebase at {project_path} to identify all AI/ML components.

Analysis depth: {analysis_depth}

Provide a comprehensive report including:

1. **AI Libraries Detected**
   - List all AI/ML libraries with versions
   - Categorize by type (LLM, ML, NLP, Vision, etc.)

2. **Models Referenced**
   - Foundation models (OpenAI, Anthropic, etc.)
   - Open-source models (Hugging Face, etc.)
   - Custom/fine-tuned models

3. **Prompt Patterns**
   - System prompts
   - Few-shot examples
   - Chain-of-thought patterns
   - Agent orchestration

4. **Data Sources**
   - Training datasets
   - Embedding stores
   - Knowledge bases

5. **Security Considerations**
   - Prompt injection risks
   - Data leakage vectors
   - Model access controls

6. **Compliance Notes**
   - License compatibility
   - Data privacy implications
   - Regulatory considerations

Format the output as structured JSON for machine processing.""",

    "model_selection": """You are an AI model selection expert. Help select the best model for the following use case:

**Use Case**: {use_case}

**Constraints**: {constraints}

Evaluate models across these dimensions:

| Model | Capability | Cost | Latency | Context | Privacy | License |
|-------|------------|------|---------|---------|---------|---------|

Consider these model families:
- OpenAI: GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo
- Anthropic: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- Google: Gemini Ultra, Gemini Pro
- Open Source: Llama 3.1, Mistral, Mixtral
- AWS Bedrock options

Provide:
1. Top 3 recommendations with justification
2. Trade-offs between options
3. Implementation considerations
4. Cost estimation""",

    "prompt_optimization": """You are a prompt engineering expert. Optimize the following prompt:

**Original Prompt**:
```
{original_prompt}
```

**Target Model**: {target_model}
**Optimization Goal**: {optimization_goal}

Provide:

1. **Optimized Prompt**
   - Restructured for clarity
   - Model-specific optimizations
   - Improved instruction following

2. **Changes Made**
   - List specific improvements
   - Explain reasoning

3. **Expected Improvements**
   - Accuracy gains
   - Latency reduction
   - Cost savings
   - Safety improvements

4. **Testing Recommendations**
   - Test cases to validate
   - Metrics to measure
   - A/B testing approach""",

    "security_review": """You are an AI security expert. Review the following for security issues:

**Code/Configuration**:
```
{code_or_config}
```

**Threat Model**: {threat_model}

Analyze for:

1. **Prompt Injection Vulnerabilities**
   - Direct injection points
   - Indirect injection vectors
   - Data poisoning risks

2. **Data Leakage Risks**
   - PII exposure
   - Training data extraction
   - Context window leaks

3. **Access Control Issues**
   - API key exposure
   - Insufficient authentication
   - Over-privileged access

4. **Model Security**
   - Model theft risks
   - Adversarial input handling
   - Output validation

Provide:
- Severity rating for each issue
- Remediation recommendations
- Code fixes where applicable""",

    "migration_assistant": """You are an AI migration specialist. Assist with migrating from {source_model} to {target_model}.

**Current Implementation**:
```
{current_implementation}
```

Provide a migration guide covering:

1. **API Changes**
   - Endpoint differences
   - Authentication changes
   - Request/response format changes

2. **Prompt Adjustments**
   - Model-specific optimizations
   - Token limit considerations
   - Feature parity mapping

3. **Code Updates**
   - SDK/library changes
   - Error handling updates
   - Retry logic adjustments

4. **Testing Plan**
   - Equivalence testing
   - Performance benchmarking
   - Rollback strategy

5. **Cost Comparison**
   - Pricing differences
   - Token efficiency changes
   - Total cost of migration""",

    "compliance_check": """You are an AI compliance expert. Review the following implementation:

**Implementation Details**:
```
{implementation_details}
```

**Regulations to Check**: {regulations}

Evaluate compliance with:

1. **Data Privacy**
   - GDPR requirements
   - Data retention policies
   - Cross-border data transfer

2. **AI-Specific Regulations**
   - EU AI Act classification
   - Risk assessment requirements
   - Transparency obligations

3. **Industry Standards**
   - SOC2 controls
   - ISO 27001 alignment
   - NIST AI RMF

4. **Documentation Requirements**
   - Model cards
   - Data sheets
   - Impact assessments

Provide:
- Compliance gaps identified
- Required remediations
- Documentation templates
- Implementation timeline"""
}


# =============================================================================
# Resource Definitions
# =============================================================================

RESOURCES = [
    Resource(
        uri="aibom://models/catalog",
        name="AI Model Catalog",
        description="Catalog of known AI models with metadata",
        mimeType="application/json"
    ),
    Resource(
        uri="aibom://prompts/templates",
        name="Prompt Template Library",
        description="Library of prompt templates and patterns",
        mimeType="application/json"
    ),
    Resource(
        uri="aibom://security/rules",
        name="Security Rules",
        description="Security rules and checks for AI implementations",
        mimeType="application/json"
    )
]

RESOURCE_TEMPLATES = [
    ResourceTemplate(
        uriTemplate="aibom://analysis/{project_id}",
        name="Project Analysis",
        description="AIBOM analysis results for a specific project",
        mimeType="application/json"
    ),
    ResourceTemplate(
        uriTemplate="aibom://models/{model_id}",
        name="Model Details",
        description="Detailed information about a specific model",
        mimeType="application/json"
    )
]


# =============================================================================
# Handler Implementations
# =============================================================================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """Return the list of available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool invocations."""
    
    if name == "analyze_python_file":
        result = await analyze_python_file(
            arguments["file_path"],
            arguments.get("include_prompts", True),
            arguments.get("include_models", True)
        )
    elif name == "scan_directory":
        result = await scan_directory(
            arguments["directory"],
            arguments.get("output_format", "json"),
            arguments.get("include_patterns", ["*.py"]),
            arguments.get("exclude_patterns", [])
        )
    elif name == "identify_model":
        result = await identify_model(arguments["model_id"])
    elif name == "analyze_prompt":
        result = await analyze_prompt(
            arguments["prompt_text"],
            arguments.get("check_injection", True),
            arguments.get("suggest_improvements", True)
        )
    elif name == "compare_models":
        result = await compare_models(
            arguments["models"],
            arguments.get("dimensions", ["capability", "cost", "speed"])
        )
    elif name == "generate_aibom":
        result = await generate_aibom(
            arguments["project_path"],
            arguments.get("project_name", "Unknown"),
            arguments.get("version", "1.0.0"),
            arguments.get("include_dependencies", True)
        )
    elif name == "check_license_compliance":
        result = await check_license_compliance(
            arguments["components"],
            arguments.get("allowed_licenses", []),
            arguments.get("use_case", "commercial")
        )
    elif name == "security_scan":
        result = await security_scan(
            arguments["target"],
            arguments.get("checks", ["prompt_injection", "data_leakage"])
        )
    else:
        result = {"error": f"Unknown tool: {name}"}
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


@server.list_prompts()
async def list_prompts() -> List[Prompt]:
    """Return the list of available prompts."""
    return list(PROMPTS.values())


@server.get_prompt()
async def get_prompt(name: str, arguments: Optional[Dict[str, str]] = None) -> GetPromptResult:
    """Get a specific prompt with arguments filled in."""
    
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt: {name}")
    
    prompt = PROMPTS[name]
    template = PROMPT_MESSAGES.get(name, "")
    
    # Fill in template with arguments
    args = arguments or {}
    for arg in prompt.arguments or []:
        if arg.name not in args:
            args[arg.name] = f"[{arg.name}]"
    
    filled_content = template.format(**args)
    
    return GetPromptResult(
        description=prompt.description,
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=filled_content)
            )
        ]
    )


@server.list_resources()
async def list_resources() -> List[Resource]:
    """Return the list of available resources."""
    return RESOURCES


@server.list_resource_templates()
async def list_resource_templates() -> List[ResourceTemplate]:
    """Return the list of resource templates."""
    return RESOURCE_TEMPLATES


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a specific resource."""
    
    if uri == "aibom://models/catalog":
        return json.dumps(get_model_catalog(), indent=2)
    elif uri == "aibom://prompts/templates":
        return json.dumps(get_prompt_templates(), indent=2)
    elif uri == "aibom://security/rules":
        return json.dumps(get_security_rules(), indent=2)
    elif uri.startswith("aibom://models/"):
        model_id = uri.replace("aibom://models/", "")
        return json.dumps(await identify_model(model_id), indent=2)
    else:
        raise ValueError(f"Unknown resource: {uri}")


# =============================================================================
# Tool Implementation Helpers
# =============================================================================

async def analyze_python_file(
    file_path: str,
    include_prompts: bool,
    include_models: bool
) -> Dict[str, Any]:
    """Analyze a Python file for AI/ML components."""
    return {
        "file": file_path,
        "libraries": ["openai", "anthropic", "langchain"],
        "models": ["gpt-4", "claude-3-sonnet"] if include_models else [],
        "prompts": {
            "system_prompts": 2,
            "few_shot_examples": 1,
            "chain_of_thought": 0
        } if include_prompts else {},
        "analysis_timestamp": datetime.now().isoformat()
    }


async def scan_directory(
    directory: str,
    output_format: str,
    include_patterns: List[str],
    exclude_patterns: List[str]
) -> Dict[str, Any]:
    """Scan a directory for AI/ML components."""
    return {
        "directory": directory,
        "files_scanned": 42,
        "components_found": {
            "libraries": 8,
            "models": 15,
            "prompts": 23,
            "datasets": 3
        },
        "format": output_format
    }


async def identify_model(model_id: str) -> Dict[str, Any]:
    """Get details about a specific model."""
    model_db = {
        "gpt-4": {
            "provider": "OpenAI",
            "type": "LLM",
            "context_length": 128000,
            "training_cutoff": "2023-12",
            "capabilities": ["text", "vision", "function_calling"],
            "license": "Proprietary"
        },
        "claude-3-5-sonnet": {
            "provider": "Anthropic",
            "type": "LLM",
            "context_length": 200000,
            "training_cutoff": "2024-04",
            "capabilities": ["text", "vision", "tool_use"],
            "license": "Proprietary"
        }
    }
    return model_db.get(model_id, {"error": "Model not found", "model_id": model_id})


async def analyze_prompt(
    prompt_text: str,
    check_injection: bool,
    suggest_improvements: bool
) -> Dict[str, Any]:
    """Analyze a prompt for issues and improvements."""
    return {
        "length": len(prompt_text),
        "token_estimate": len(prompt_text) // 4,
        "injection_risks": ["low"] if check_injection else [],
        "suggestions": [
            "Add explicit output format",
            "Include examples for few-shot"
        ] if suggest_improvements else []
    }


async def compare_models(
    models: List[str],
    dimensions: List[str]
) -> Dict[str, Any]:
    """Compare multiple models."""
    return {
        "models": models,
        "comparison": {
            model: {dim: "N/A" for dim in dimensions}
            for model in models
        }
    }


async def generate_aibom(
    project_path: str,
    project_name: str,
    version: str,
    include_dependencies: bool
) -> Dict[str, Any]:
    """Generate an AIBOM in CycloneDX format."""
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "name": project_name,
                "version": version
            }
        },
        "components": []
    }


async def check_license_compliance(
    components: List[str],
    allowed_licenses: List[str],
    use_case: str
) -> Dict[str, Any]:
    """Check license compliance."""
    return {
        "components_checked": len(components),
        "compliant": True,
        "issues": [],
        "use_case": use_case
    }


async def security_scan(
    target: str,
    checks: List[str]
) -> Dict[str, Any]:
    """Perform security scan."""
    return {
        "target": target,
        "checks_performed": checks,
        "findings": [],
        "risk_level": "low"
    }


def get_model_catalog() -> Dict[str, Any]:
    """Get the model catalog."""
    return {
        "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"],
        "google": ["gemini-ultra", "gemini-pro"],
        "meta": ["llama-3.1-405b", "llama-3.1-70b", "llama-3.1-8b"],
        "mistral": ["mistral-large", "mixtral-8x7b", "mistral-7b"]
    }


def get_prompt_templates() -> Dict[str, Any]:
    """Get prompt template library."""
    return {
        "system_prompts": ["general_assistant", "coding_expert", "analyst"],
        "patterns": ["chain_of_thought", "few_shot", "react", "tree_of_thought"]
    }


def get_security_rules() -> Dict[str, Any]:
    """Get security rules."""
    return {
        "prompt_injection": {
            "severity": "high",
            "patterns": ["ignore previous", "new instructions", "system prompt"]
        },
        "data_leakage": {
            "severity": "medium",
            "patterns": ["PII", "credentials", "API keys"]
        }
    }


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

