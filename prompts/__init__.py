"""
Prompt Templates and Examples for AIBOM Detection Testing

This package contains various prompt patterns used with AI models:
- System prompts for different use cases
- Few-shot learning examples
- Chain-of-thought reasoning patterns
- Tool use and function calling patterns
- Structured output formats
- Security and safety prompts
- Agentic workflow prompts
- Model-specific optimizations
- Evaluation and benchmark prompts

These prompts are designed to test whether AIBOM scanners can detect
prompt engineering artifacts as part of an AI Bill of Materials.
"""

from .system_prompts import (
    SYSTEM_PROMPTS,
    get_system_prompt,
    GENERAL_ASSISTANT_PROMPT,
    CODING_ASSISTANT_PROMPT,
    RAG_SYSTEM_PROMPT,
    AGENT_SYSTEM_PROMPT,
)

from .few_shot_examples import (
    FEW_SHOT_PROMPTS,
    SENTIMENT_FEW_SHOT_PROMPT,
    NER_FEW_SHOT_PROMPT,
    CODE_FEW_SHOT_PROMPT,
    create_few_shot_prompt,
)

from .chain_of_thought import (
    COT_PROMPTS,
    get_cot_prompt,
    BASIC_COT_PROMPT,
    MATH_COT_PROMPT,
    TREE_OF_THOUGHT_PROMPT,
)

from .tool_use_prompts import (
    OPENAI_TOOLS_DEFINITION,
    ANTHROPIC_TOOLS_DEFINITION,
    BEDROCK_ACTION_GROUP_SCHEMA,
    MCP_TOOLS,
    TOOL_REGISTRIES,
)

from .structured_output import (
    STRUCTURED_OUTPUT_PROMPTS,
    OUTPUT_SCHEMAS,
    get_schema_json,
    validate_output,
)

from .security_prompts import (
    SECURITY_PROMPTS,
    PROMPT_INJECTION_EXAMPLES,
    ADVERSARIAL_TEST_CASES,
    REFUSAL_TEMPLATES,
    generate_injection_test_suite,
)

from .agentic_prompts import (
    AGENTIC_PROMPTS,
    create_agent_system,
    AUTONOMOUS_AGENT_SYSTEM,
    COORDINATOR_AGENT,
    REFLECTION_AGENT,
)

from .model_specific_prompts import (
    MODEL_PROMPTS,
    get_model_prompt,
    GPT4_SYSTEM_PROMPT,
    CLAUDE_3_SYSTEM_PROMPT,
    LLAMA_3_SYSTEM_PROMPT,
    GEMINI_SYSTEM_PROMPT,
    BEDROCK_CLAUDE_PROMPT,
)

from .evaluation_prompts import (
    EVALUATION_PROMPTS,
    get_eval_prompt,
    MMLU_PROMPT_TEMPLATE,
    HUMANEVAL_PROMPT,
    RED_TEAM_EVALUATION,
)

__all__ = [
    # System prompts
    "SYSTEM_PROMPTS",
    "get_system_prompt",
    "GENERAL_ASSISTANT_PROMPT",
    "CODING_ASSISTANT_PROMPT",
    "RAG_SYSTEM_PROMPT",
    "AGENT_SYSTEM_PROMPT",
    # Few-shot
    "FEW_SHOT_PROMPTS",
    "SENTIMENT_FEW_SHOT_PROMPT",
    "NER_FEW_SHOT_PROMPT",
    "CODE_FEW_SHOT_PROMPT",
    "create_few_shot_prompt",
    # Chain of thought
    "COT_PROMPTS",
    "get_cot_prompt",
    "BASIC_COT_PROMPT",
    "MATH_COT_PROMPT",
    "TREE_OF_THOUGHT_PROMPT",
    # Tool use
    "OPENAI_TOOLS_DEFINITION",
    "ANTHROPIC_TOOLS_DEFINITION",
    "BEDROCK_ACTION_GROUP_SCHEMA",
    "MCP_TOOLS",
    "TOOL_REGISTRIES",
    # Structured output
    "STRUCTURED_OUTPUT_PROMPTS",
    "OUTPUT_SCHEMAS",
    "get_schema_json",
    "validate_output",
    # Security
    "SECURITY_PROMPTS",
    "PROMPT_INJECTION_EXAMPLES",
    "ADVERSARIAL_TEST_CASES",
    "REFUSAL_TEMPLATES",
    "generate_injection_test_suite",
    # Agentic
    "AGENTIC_PROMPTS",
    "create_agent_system",
    "AUTONOMOUS_AGENT_SYSTEM",
    "COORDINATOR_AGENT",
    "REFLECTION_AGENT",
    # Model-specific
    "MODEL_PROMPTS",
    "get_model_prompt",
    "GPT4_SYSTEM_PROMPT",
    "CLAUDE_3_SYSTEM_PROMPT",
    "LLAMA_3_SYSTEM_PROMPT",
    "GEMINI_SYSTEM_PROMPT",
    "BEDROCK_CLAUDE_PROMPT",
    # Evaluation
    "EVALUATION_PROMPTS",
    "get_eval_prompt",
    "MMLU_PROMPT_TEMPLATE",
    "HUMANEVAL_PROMPT",
    "RED_TEAM_EVALUATION",
]
