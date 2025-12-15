"""
Structured Output Prompt Examples
Testing if AIBOM can detect JSON/schema output patterns.
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import json

# =============================================================================
# PYDANTIC OUTPUT SCHEMAS
# =============================================================================

class EntityExtraction(BaseModel):
    """Schema for extracted entities."""
    entities: List[Dict[str, str]] = Field(description="List of extracted entities")
    confidence: float = Field(ge=0, le=1, description="Extraction confidence")
    raw_text: str = Field(description="Original input text")


class SentimentResult(BaseModel):
    """Schema for sentiment analysis."""
    sentiment: str = Field(description="positive, negative, neutral, or mixed")
    confidence: float = Field(ge=0, le=1)
    aspects: List[Dict[str, str]] = Field(default=[], description="Aspect-based sentiments")


class CodeAnalysis(BaseModel):
    """Schema for code analysis results."""
    language: str
    complexity_score: int = Field(ge=0, le=100)
    issues: List[Dict[str, Any]] = Field(default=[])
    suggestions: List[str] = Field(default=[])
    security_risks: List[str] = Field(default=[])


class DocumentSummary(BaseModel):
    """Schema for document summarization."""
    title: Optional[str] = None
    summary: str
    key_points: List[str]
    word_count: int
    reading_time_minutes: int


class ClassificationResult(BaseModel):
    """Schema for multi-class classification."""
    primary_class: str
    confidence: float
    all_classes: List[Dict[str, float]] = Field(description="All classes with probabilities")


# =============================================================================
# JSON OUTPUT PROMPTS
# =============================================================================

JSON_OUTPUT_PROMPT = """Analyze the following text and respond with valid JSON only.

Text: {text}

Respond with this exact JSON structure:
{{
    "summary": "brief summary",
    "key_entities": ["entity1", "entity2"],
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0
}}

JSON Response:"""

STRICT_JSON_PROMPT = """You must respond with valid JSON that matches this schema exactly.

Schema:
{schema}

Input: {input}

Important: 
- Output ONLY valid JSON
- No markdown code blocks
- No explanations before or after
- Match the schema exactly

JSON:"""

# =============================================================================
# STRUCTURED EXTRACTION PROMPTS
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """Extract all named entities from the text.

Text: {text}

Return a JSON object with this structure:
{{
    "persons": ["list of person names"],
    "organizations": ["list of organization names"],
    "locations": ["list of location names"],
    "dates": ["list of dates"],
    "monetary_values": ["list of money amounts"],
    "percentages": ["list of percentages"],
    "other": ["other notable entities"]
}}

Extracted Entities:"""

RELATIONSHIP_EXTRACTION_PROMPT = """Extract relationships between entities in the text.

Text: {text}

Return JSON with this structure:
{{
    "entities": [
        {{"id": 1, "text": "entity text", "type": "PERSON|ORG|LOCATION|etc"}}
    ],
    "relationships": [
        {{"source_id": 1, "target_id": 2, "relationship": "works_for|located_in|etc"}}
    ]
}}

Extraction:"""

TABLE_EXTRACTION_PROMPT = """Extract tabular data from the text.

Text: {text}

Return the data as a JSON array of objects:
{{
    "table_name": "descriptive name",
    "columns": ["col1", "col2", "col3"],
    "rows": [
        {{"col1": "value1", "col2": "value2", "col3": "value3"}},
        ...
    ]
}}

Extracted Table:"""

# =============================================================================
# API RESPONSE FORMAT PROMPTS
# =============================================================================

API_RESPONSE_PROMPT = """Generate an API response for the given request.

Request: {request}

Respond with a properly formatted API response:
{{
    "status": "success|error",
    "code": 200,
    "data": {{}},
    "meta": {{
        "timestamp": "ISO8601",
        "request_id": "uuid"
    }}
}}

Response:"""

ERROR_RESPONSE_PROMPT = """Generate an appropriate error response.

Error scenario: {error}

{{
    "status": "error",
    "error": {{
        "code": "ERROR_CODE",
        "message": "Human-readable message",
        "details": {{}},
        "suggestion": "How to fix"
    }}
}}

Error Response:"""

# =============================================================================
# FORM/DATA EXTRACTION PROMPTS
# =============================================================================

FORM_EXTRACTION_PROMPT = """Extract form field data from this document/image.

Content: {content}

Return structured form data:
{{
    "form_type": "invoice|receipt|application|etc",
    "fields": {{
        "field_name": {{
            "value": "extracted value",
            "confidence": 0.95,
            "bounding_box": [x, y, width, height]  // if applicable
        }}
    }},
    "validation_issues": []
}}

Extracted Form Data:"""

INVOICE_EXTRACTION_PROMPT = """Extract invoice information.

Invoice content: {content}

{{
    "invoice_number": "",
    "date": "",
    "due_date": "",
    "vendor": {{
        "name": "",
        "address": "",
        "tax_id": ""
    }},
    "customer": {{
        "name": "",
        "address": ""
    }},
    "line_items": [
        {{
            "description": "",
            "quantity": 0,
            "unit_price": 0.00,
            "total": 0.00
        }}
    ],
    "subtotal": 0.00,
    "tax": 0.00,
    "total": 0.00,
    "currency": "USD"
}}

Extracted Invoice:"""

# =============================================================================
# CODE GENERATION WITH SCHEMA
# =============================================================================

CODE_WITH_SCHEMA_PROMPT = """Generate code that matches this specification.

Specification:
{spec}

Output format:
{{
    "code": "the generated code",
    "language": "python|javascript|etc",
    "dependencies": ["list", "of", "imports"],
    "usage_example": "example of how to use",
    "tests": ["test case 1", "test case 2"]
}}

Generated Code:"""

# =============================================================================
# EVALUATION/SCORING PROMPTS
# =============================================================================

EVALUATION_PROMPT = """Evaluate the following content on multiple criteria.

Content: {content}
Criteria: {criteria}

Return evaluation scores:
{{
    "overall_score": 0-100,
    "criteria_scores": {{
        "criterion_name": {{
            "score": 0-100,
            "explanation": "why this score",
            "suggestions": ["improvement suggestion"]
        }}
    }},
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "recommendation": "overall recommendation"
}}

Evaluation:"""

COMPARISON_PROMPT = """Compare these two items.

Item A: {item_a}
Item B: {item_b}

{{
    "winner": "A|B|tie",
    "comparison": {{
        "aspect": {{
            "item_a_score": 0-10,
            "item_b_score": 0-10,
            "explanation": "why"
        }}
    }},
    "summary": "overall comparison summary"
}}

Comparison Result:"""

# =============================================================================
# STRUCTURED OUTPUT REGISTRY
# =============================================================================

STRUCTURED_OUTPUT_PROMPTS = {
    "json_basic": JSON_OUTPUT_PROMPT,
    "json_strict": STRICT_JSON_PROMPT,
    "entity_extraction": ENTITY_EXTRACTION_PROMPT,
    "relationship_extraction": RELATIONSHIP_EXTRACTION_PROMPT,
    "table_extraction": TABLE_EXTRACTION_PROMPT,
    "api_response": API_RESPONSE_PROMPT,
    "error_response": ERROR_RESPONSE_PROMPT,
    "form_extraction": FORM_EXTRACTION_PROMPT,
    "invoice_extraction": INVOICE_EXTRACTION_PROMPT,
    "code_with_schema": CODE_WITH_SCHEMA_PROMPT,
    "evaluation": EVALUATION_PROMPT,
    "comparison": COMPARISON_PROMPT,
}

OUTPUT_SCHEMAS = {
    "entity_extraction": EntityExtraction,
    "sentiment": SentimentResult,
    "code_analysis": CodeAnalysis,
    "document_summary": DocumentSummary,
    "classification": ClassificationResult,
}


def get_schema_json(schema_name: str) -> str:
    """Get JSON schema for a Pydantic model."""
    schema_class = OUTPUT_SCHEMAS.get(schema_name)
    if schema_class:
        return json.dumps(schema_class.model_json_schema(), indent=2)
    return "{}"


def validate_output(output: str, schema_name: str) -> tuple[bool, Any]:
    """Validate JSON output against a schema."""
    schema_class = OUTPUT_SCHEMAS.get(schema_name)
    if not schema_class:
        return False, "Unknown schema"
    
    try:
        data = json.loads(output)
        validated = schema_class(**data)
        return True, validated
    except Exception as e:
        return False, str(e)

