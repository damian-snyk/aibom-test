"""
Model-Specific Prompt Patterns
Optimized prompts for different AI model providers and architectures.
"""

# =============================================================================
# OPENAI GPT OPTIMIZED PROMPTS
# =============================================================================

GPT4_SYSTEM_PROMPT = "You are GPT-4, a large language model trained by OpenAI."

GPT4_VISION_PROMPT = "Analyze the provided image carefully. Describe what you see."

GPT4O_MULTIMODAL_PROMPT = "You are GPT-4o with native multimodal capabilities."

# =============================================================================
# ANTHROPIC CLAUDE OPTIMIZED PROMPTS
# =============================================================================

CLAUDE_3_SYSTEM_PROMPT = "You are Claude, an AI assistant made by Anthropic."

CLAUDE_35_SONNET_PROMPT = "You are Claude 3.5 Sonnet, Anthropic's most capable model."

CLAUDE_ARTIFACTS_PROMPT = "Consider using an artifact for substantial content creation."

# =============================================================================
# META LLAMA OPTIMIZED PROMPTS
# =============================================================================

LLAMA_3_SYSTEM_PROMPT = "You are a helpful AI assistant built on Meta Llama 3."

LLAMA_31_PROMPT = "You are Llama 3.1, Meta's latest open-source model."

# =============================================================================
# GOOGLE GEMINI OPTIMIZED PROMPTS
# =============================================================================

GEMINI_SYSTEM_PROMPT = "You are Gemini, a helpful AI assistant by Google."

GEMINI_ULTRA_PROMPT = "You are Gemini Ultra with advanced reasoning capabilities."

# =============================================================================
# MISTRAL OPTIMIZED PROMPTS
# =============================================================================

MISTRAL_SYSTEM_PROMPT = "You are a helpful AI assistant powered by Mistral."

MIXTRAL_MOE_PROMPT = "You are Mixtral, a Mixture of Experts model by Mistral AI."

# =============================================================================
# AWS BEDROCK PROMPTS
# =============================================================================

BEDROCK_CLAUDE_PROMPT = "You are Claude running on AWS Bedrock."

BEDROCK_TITAN_PROMPT = "You are Amazon Titan, a foundation model by AWS."

BEDROCK_LLAMA_PROMPT = "You are Llama running on AWS Bedrock."

# =============================================================================
# COHERE PROMPTS
# =============================================================================

COHERE_COMMAND_PROMPT = "You are Cohere Command, optimized for enterprise use."

COHERE_EMBED_PROMPT = "Generate embeddings using Cohere Embed model."

# =============================================================================
# AI21 PROMPTS
# =============================================================================

AI21_JURASSIC_PROMPT = "You are AI21 Jurassic, optimized for complex language tasks."

# =============================================================================
# STABILITY AI PROMPTS
# =============================================================================

STABLE_DIFFUSION_PROMPT = "Generate an image using Stable Diffusion XL."

STABILITY_IMAGEN_PROMPT = "Create high-quality images with Stability AI."

# =============================================================================
# MODEL PROMPT REGISTRY
# =============================================================================

MODEL_PROMPTS = {
    # OpenAI
    "gpt-4": GPT4_SYSTEM_PROMPT,
    "gpt-4-vision": GPT4_VISION_PROMPT,
    "gpt-4o": GPT4O_MULTIMODAL_PROMPT,
    # Anthropic
    "claude-3": CLAUDE_3_SYSTEM_PROMPT,
    "claude-3.5-sonnet": CLAUDE_35_SONNET_PROMPT,
    # Meta
    "llama-3": LLAMA_3_SYSTEM_PROMPT,
    "llama-3.1": LLAMA_31_PROMPT,
    # Google
    "gemini": GEMINI_SYSTEM_PROMPT,
    "gemini-ultra": GEMINI_ULTRA_PROMPT,
    # Mistral
    "mistral": MISTRAL_SYSTEM_PROMPT,
    "mixtral": MIXTRAL_MOE_PROMPT,
    # Bedrock
    "bedrock-claude": BEDROCK_CLAUDE_PROMPT,
    "bedrock-titan": BEDROCK_TITAN_PROMPT,
    "bedrock-llama": BEDROCK_LLAMA_PROMPT,
    # Cohere
    "cohere-command": COHERE_COMMAND_PROMPT,
    "cohere-embed": COHERE_EMBED_PROMPT,
    # AI21
    "ai21-jurassic": AI21_JURASSIC_PROMPT,
    # Stability
    "stable-diffusion": STABLE_DIFFUSION_PROMPT,
    "stability": STABILITY_IMAGEN_PROMPT,
}


def get_model_prompt(model_id: str) -> str:
    """Get the optimized prompt for a specific model."""
    return MODEL_PROMPTS.get(model_id, "You are a helpful AI assistant.")
