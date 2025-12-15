"""
MCP Server Prompt Templates

This module contains prompt templates that are served by the MCP server.
These prompts are designed for AI assistant interactions through the
Model Context Protocol.
"""

from typing import Dict, List, Any

# =============================================================================
# AIBOM Analysis Prompts
# =============================================================================

AIBOM_QUICK_SCAN_PROMPT = """Perform a quick AIBOM scan of the project.

Project: {project_path}

Focus on:
1. Identifying main AI/ML libraries
2. Finding explicit model references
3. Detecting obvious prompt patterns

Output a brief summary suitable for a status check."""

AIBOM_DEEP_SCAN_PROMPT = """Perform a comprehensive AIBOM analysis.

Project: {project_path}

Analyze:

## 1. Dependency Analysis
- Direct AI/ML library imports
- Transitive dependencies
- Version constraints and compatibility

## 2. Model Inventory
- Foundation models (by provider)
- Open-source models (with checksums if available)
- Custom/fine-tuned models
- Model configuration files

## 3. Prompt Engineering Artifacts
- System prompts and their purposes
- Few-shot example collections
- Prompt templates and variables
- Chain-of-thought patterns
- Agent orchestration prompts
- Tool/function definitions

## 4. Data Assets
- Training datasets
- Evaluation datasets
- Vector stores and embeddings
- Knowledge bases

## 5. Infrastructure
- Model serving configurations
- API integrations
- Caching layers
- Rate limiting configs

## 6. Security Analysis
- API key management
- Prompt injection vectors
- Data leakage risks
- Access control patterns

Generate a detailed report in JSON format following CycloneDX 1.6 schema."""

AIBOM_DIFF_PROMPT = """Compare AIBOM between two versions.

Previous AIBOM: {previous_aibom}
Current AIBOM: {current_aibom}

Identify:
1. Added components
2. Removed components
3. Version changes
4. Configuration changes
5. Security implications of changes

Generate a change report suitable for PR review."""

# =============================================================================
# Model Analysis Prompts
# =============================================================================

MODEL_EVALUATION_PROMPT = """Evaluate AI model for production readiness.

Model: {model_id}
Use Case: {use_case}
Environment: {environment}

Assess:

## Capability Assessment
- Task suitability (1-10)
- Output quality expectations
- Edge case handling
- Multilingual support

## Operational Considerations
- Latency requirements
- Throughput needs
- Availability SLA
- Failover strategy

## Cost Analysis
- Per-token pricing
- Expected monthly volume
- Cost optimization strategies

## Risk Assessment
- Hallucination risk
- Bias considerations
- Data privacy
- Regulatory compliance

## Integration Requirements
- API compatibility
- SDK availability
- Monitoring capabilities
- Logging requirements

Provide a go/no-go recommendation with justification."""

MODEL_COMPARISON_PROMPT = """Compare AI models for selection.

Models to compare: {models}
Selection criteria: {criteria}
Use case: {use_case}

Create a detailed comparison matrix:

| Criterion | {model_headers} | Winner |
|-----------|{model_cols}|--------|

For each criterion, provide:
- Quantitative score (1-10)
- Qualitative assessment
- Supporting evidence

Final recommendation with:
- Primary choice
- Backup option
- Migration path between them"""

MODEL_MIGRATION_PROMPT = """Plan model migration.

Source: {source_model}
Target: {target_model}
Current usage: {current_usage}

Migration Plan:

## 1. Compatibility Analysis
- API differences
- Feature parity
- Breaking changes

## 2. Prompt Adaptation
- Required modifications
- Testing strategy
- Rollback triggers

## 3. Implementation Steps
- Development changes
- Configuration updates
- Deployment sequence

## 4. Validation
- Equivalence testing
- Performance benchmarks
- User acceptance criteria

## 5. Rollout Strategy
- Canary deployment
- A/B testing
- Full migration
- Rollback procedure

## 6. Monitoring
- Key metrics
- Alert thresholds
- Success criteria

Provide timeline and resource estimates."""

# =============================================================================
# Security Review Prompts
# =============================================================================

SECURITY_PROMPT_INJECTION_REVIEW = """Review code for prompt injection vulnerabilities.

Code:
```
{code}
```

Analyze for:

## Direct Injection Vectors
- User input concatenation
- Template injection
- Delimiter confusion

## Indirect Injection Vectors
- Data source contamination
- Cross-context attacks
- Persistent injection

## Mitigation Assessment
- Input validation present?
- Output filtering?
- Context isolation?

For each vulnerability found:
1. Location in code
2. Attack scenario
3. Severity (Critical/High/Medium/Low)
4. Remediation code

Overall risk score: [X/10]"""

SECURITY_DATA_LEAKAGE_REVIEW = """Review for AI-related data leakage risks.

Implementation:
```
{implementation}
```

Check for:

## Training Data Exposure
- Memorization attacks
- Extraction attempts
- Fingerprinting

## Runtime Data Leakage
- Prompt logging
- Response caching
- Error messages

## PII Handling
- Detection mechanisms
- Anonymization
- Retention policies

## Context Window Risks
- Session isolation
- Context overflow
- History management

Provide:
- Risk matrix
- Compliance gaps
- Remediation priority list"""

SECURITY_ACCESS_CONTROL_REVIEW = """Review AI access control implementation.

Configuration:
```
{config}
```

Evaluate:

## Authentication
- API key security
- Token management
- Credential rotation

## Authorization
- Role-based access
- Resource-level permissions
- Rate limiting

## Audit
- Request logging
- Usage tracking
- Anomaly detection

## Secrets Management
- Storage security
- Environment isolation
- Rotation policies

Security score: [X/10]
Priority fixes: [list]"""

# =============================================================================
# Compliance Prompts
# =============================================================================

COMPLIANCE_EU_AI_ACT_PROMPT = """Assess EU AI Act compliance.

System description:
{system_description}

Evaluate:

## Risk Classification
- Unacceptable risk
- High risk
- Limited risk
- Minimal risk

## Transparency Requirements
- Disclosure obligations
- User notification
- Documentation

## Data Governance
- Training data requirements
- Bias mitigation
- Human oversight

## Technical Requirements
- Accuracy metrics
- Robustness testing
- Cybersecurity

## Documentation
- Technical documentation
- Conformity assessment
- CE marking (if applicable)

Compliance status: [Compliant/Gaps Identified/Non-Compliant]
Required actions: [prioritized list]"""

COMPLIANCE_SOC2_AI_PROMPT = """Assess SOC2 compliance for AI systems.

AI Implementation:
{implementation}

Map to Trust Service Criteria:

## Security
- CC6: Logical and Physical Access Controls
- CC7: System Operations
- CC8: Change Management

## Availability
- A1: System availability commitments

## Processing Integrity
- PI1: Quality of processing

## Confidentiality
- C1: Protection of confidential information

## Privacy
- P1-P8: Privacy criteria

For AI-specific controls, assess:
- Model versioning
- Prompt management
- Output monitoring
- Incident response

Control gaps: [list]
Remediation roadmap: [timeline]"""

# =============================================================================
# Development Prompts
# =============================================================================

CODE_REVIEW_AI_PROMPT = """Review AI/ML code for best practices.

Code:
```{language}
{code}
```

Evaluate:

## API Usage
- Proper error handling
- Retry logic
- Rate limit handling
- Timeout configuration

## Prompt Engineering
- Clarity and specificity
- Token efficiency
- Output format specification
- Edge case handling

## Data Handling
- Input validation
- Output sanitization
- Logging practices
- Caching strategy

## Observability
- Metrics collection
- Tracing
- Cost tracking
- Quality monitoring

## Testing
- Unit test coverage
- Mock strategies
- Evaluation datasets
- Regression tests

Provide:
- Code quality score: [X/10]
- Critical issues: [list]
- Improvement suggestions: [list]
- Refactored code snippets where applicable"""

PROMPT_ENGINEERING_REVIEW = """Review and optimize prompt.

Original Prompt:
```
{prompt}
```

Target Model: {model}
Task: {task}

Analysis:

## Structure Assessment
- Clarity of instructions
- Role definition
- Context provision
- Output specification

## Token Efficiency
- Current token count
- Redundancy identification
- Compression opportunities

## Robustness
- Edge case handling
- Ambiguity resolution
- Error recovery

## Model Optimization
- Model-specific patterns
- Capability utilization
- Limitation awareness

Optimized Prompt:
```
[optimized version]
```

Changes made:
1. [change 1]
2. [change 2]
...

Expected improvements:
- Accuracy: +X%
- Token reduction: -Y%
- Consistency: +Z%"""

# =============================================================================
# Prompt Template Registry
# =============================================================================

MCP_PROMPT_REGISTRY = {
    # AIBOM
    "aibom_quick_scan": AIBOM_QUICK_SCAN_PROMPT,
    "aibom_deep_scan": AIBOM_DEEP_SCAN_PROMPT,
    "aibom_diff": AIBOM_DIFF_PROMPT,
    # Model Analysis
    "model_evaluation": MODEL_EVALUATION_PROMPT,
    "model_comparison": MODEL_COMPARISON_PROMPT,
    "model_migration": MODEL_MIGRATION_PROMPT,
    # Security
    "security_prompt_injection": SECURITY_PROMPT_INJECTION_REVIEW,
    "security_data_leakage": SECURITY_DATA_LEAKAGE_REVIEW,
    "security_access_control": SECURITY_ACCESS_CONTROL_REVIEW,
    # Compliance
    "compliance_eu_ai_act": COMPLIANCE_EU_AI_ACT_PROMPT,
    "compliance_soc2_ai": COMPLIANCE_SOC2_AI_PROMPT,
    # Development
    "code_review_ai": CODE_REVIEW_AI_PROMPT,
    "prompt_engineering_review": PROMPT_ENGINEERING_REVIEW,
}


def get_mcp_prompt(name: str, **kwargs) -> str:
    """Get an MCP prompt template with variables filled in."""
    template = MCP_PROMPT_REGISTRY.get(name)
    if not template:
        raise ValueError(f"Unknown prompt: {name}")
    
    # Fill in provided kwargs, leave others as placeholders
    try:
        return template.format(**kwargs)
    except KeyError as e:
        # Return template with unfilled variables marked
        return template


def list_mcp_prompts() -> List[Dict[str, str]]:
    """List all available MCP prompts with descriptions."""
    descriptions = {
        "aibom_quick_scan": "Quick scan for AI/ML components",
        "aibom_deep_scan": "Comprehensive AIBOM analysis",
        "aibom_diff": "Compare AIBOM between versions",
        "model_evaluation": "Evaluate model for production",
        "model_comparison": "Compare multiple models",
        "model_migration": "Plan model migration",
        "security_prompt_injection": "Review for prompt injection",
        "security_data_leakage": "Review for data leakage",
        "security_access_control": "Review access controls",
        "compliance_eu_ai_act": "EU AI Act compliance check",
        "compliance_soc2_ai": "SOC2 AI controls assessment",
        "code_review_ai": "AI code review",
        "prompt_engineering_review": "Prompt optimization review",
    }
    
    return [
        {"name": name, "description": descriptions.get(name, "")}
        for name in MCP_PROMPT_REGISTRY.keys()
    ]

