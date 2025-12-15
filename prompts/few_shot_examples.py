"""
Few-Shot Learning Prompt Examples
Testing if AIBOM can detect few-shot prompting patterns.
"""
from typing import List, Dict

# =============================================================================
# SENTIMENT ANALYSIS FEW-SHOT EXAMPLES
# =============================================================================

SENTIMENT_ANALYSIS_EXAMPLES = [
    {
        "input": "This product exceeded all my expectations! Absolutely love it.",
        "output": "positive",
        "confidence": 0.95
    },
    {
        "input": "Terrible experience. Would not recommend to anyone.",
        "output": "negative",
        "confidence": 0.92
    },
    {
        "input": "It's okay, nothing special but gets the job done.",
        "output": "neutral",
        "confidence": 0.78
    },
    {
        "input": "The service was good but the product quality was disappointing.",
        "output": "mixed",
        "confidence": 0.65
    }
]

SENTIMENT_FEW_SHOT_PROMPT = """Classify the sentiment of the following text.

Examples:
Text: "This product exceeded all my expectations! Absolutely love it."
Sentiment: positive

Text: "Terrible experience. Would not recommend to anyone."
Sentiment: negative

Text: "It's okay, nothing special but gets the job done."
Sentiment: neutral

Text: "The service was good but the product quality was disappointing."
Sentiment: mixed

Now classify this text:
Text: "{input_text}"
Sentiment:"""

# =============================================================================
# NAMED ENTITY RECOGNITION FEW-SHOT EXAMPLES
# =============================================================================

NER_EXAMPLES = [
    {
        "input": "Apple CEO Tim Cook announced the new iPhone at their Cupertino headquarters.",
        "output": {
            "ORGANIZATION": ["Apple"],
            "PERSON": ["Tim Cook"],
            "PRODUCT": ["iPhone"],
            "LOCATION": ["Cupertino"]
        }
    },
    {
        "input": "Elon Musk's Tesla delivered 500,000 vehicles in Q4 2023.",
        "output": {
            "PERSON": ["Elon Musk"],
            "ORGANIZATION": ["Tesla"],
            "QUANTITY": ["500,000"],
            "DATE": ["Q4 2023"]
        }
    },
    {
        "input": "The European Union fined Google €4.3 billion in Brussels.",
        "output": {
            "ORGANIZATION": ["European Union", "Google"],
            "MONEY": ["€4.3 billion"],
            "LOCATION": ["Brussels"]
        }
    }
]

NER_FEW_SHOT_PROMPT = """Extract named entities from the text.

Example 1:
Text: "Apple CEO Tim Cook announced the new iPhone at their Cupertino headquarters."
Entities:
- ORGANIZATION: Apple
- PERSON: Tim Cook
- PRODUCT: iPhone
- LOCATION: Cupertino

Example 2:
Text: "Elon Musk's Tesla delivered 500,000 vehicles in Q4 2023."
Entities:
- PERSON: Elon Musk
- ORGANIZATION: Tesla
- QUANTITY: 500,000
- DATE: Q4 2023

Now extract entities from:
Text: "{input_text}"
Entities:"""

# =============================================================================
# CODE GENERATION FEW-SHOT EXAMPLES
# =============================================================================

CODE_GENERATION_EXAMPLES = [
    {
        "instruction": "Write a function to check if a number is prime",
        "language": "python",
        "code": '''def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True'''
    },
    {
        "instruction": "Write a function to reverse a string",
        "language": "python",
        "code": '''def reverse_string(s: str) -> str:
    """Reverse a string."""
    return s[::-1]'''
    },
    {
        "instruction": "Write a function to find the factorial of a number",
        "language": "python",
        "code": '''def factorial(n: int) -> int:
    """Calculate factorial recursively."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)'''
    }
]

CODE_FEW_SHOT_PROMPT = '''Generate Python code based on the instruction.

### Instruction: Write a function to check if a number is prime
### Code:
```python
def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```

### Instruction: Write a function to reverse a string
### Code:
```python
def reverse_string(s: str) -> str:
    """Reverse a string."""
    return s[::-1]
```

### Instruction: {instruction}
### Code:
```python'''

# =============================================================================
# SQL GENERATION FEW-SHOT EXAMPLES
# =============================================================================

SQL_GENERATION_EXAMPLES = [
    {
        "question": "How many users signed up last month?",
        "schema": "users(id, name, email, created_at)",
        "sql": "SELECT COUNT(*) FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"
    },
    {
        "question": "What are the top 5 products by revenue?",
        "schema": "products(id, name), orders(id, product_id, amount)",
        "sql": "SELECT p.name, SUM(o.amount) as revenue FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.id ORDER BY revenue DESC LIMIT 5"
    },
    {
        "question": "Find users who haven't made a purchase",
        "schema": "users(id, name), orders(id, user_id)",
        "sql": "SELECT u.* FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.id IS NULL"
    }
]

SQL_FEW_SHOT_PROMPT = """Generate SQL queries based on natural language questions.

Schema: users(id, name, email, created_at)
Question: How many users signed up last month?
SQL: SELECT COUNT(*) FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH)

Schema: products(id, name), orders(id, product_id, amount)
Question: What are the top 5 products by revenue?
SQL: SELECT p.name, SUM(o.amount) as revenue FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.id ORDER BY revenue DESC LIMIT 5

Schema: {schema}
Question: {question}
SQL:"""

# =============================================================================
# CLASSIFICATION FEW-SHOT EXAMPLES
# =============================================================================

INTENT_CLASSIFICATION_EXAMPLES = [
    {"text": "I want to book a flight to New York", "intent": "travel_booking"},
    {"text": "What's the weather like today?", "intent": "weather_query"},
    {"text": "Cancel my subscription", "intent": "account_cancellation"},
    {"text": "I need help resetting my password", "intent": "password_reset"},
    {"text": "Tell me a joke", "intent": "entertainment"},
    {"text": "Schedule a meeting for tomorrow", "intent": "calendar_management"},
]

INTENT_FEW_SHOT_PROMPT = """Classify the user's intent.

User: "I want to book a flight to New York"
Intent: travel_booking

User: "What's the weather like today?"
Intent: weather_query

User: "Cancel my subscription"
Intent: account_cancellation

User: "I need help resetting my password"
Intent: password_reset

User: "{user_input}"
Intent:"""

# =============================================================================
# SUMMARIZATION FEW-SHOT EXAMPLES
# =============================================================================

SUMMARIZATION_EXAMPLES = [
    {
        "document": "The quarterly earnings report shows a 15% increase in revenue compared to the same period last year. Net income grew by 22%, driven primarily by cost reduction initiatives and expansion into new markets. The company also announced plans to hire 500 new employees in the coming year.",
        "summary": "Company reports 15% revenue growth and 22% net income increase YoY, with plans to hire 500 employees."
    },
    {
        "document": "A new study published in Nature reveals that regular exercise can reduce the risk of heart disease by up to 40%. Researchers followed 10,000 participants over 15 years and found that those who exercised at least 30 minutes daily had significantly better cardiovascular health outcomes.",
        "summary": "15-year study of 10,000 participants shows daily 30-minute exercise reduces heart disease risk by 40%."
    }
]

SUMMARIZATION_FEW_SHOT_PROMPT = """Summarize the following document in one sentence.

Document: "The quarterly earnings report shows a 15% increase in revenue compared to the same period last year. Net income grew by 22%, driven primarily by cost reduction initiatives and expansion into new markets."
Summary: Company reports 15% revenue growth and 22% net income increase YoY due to cost cuts and market expansion.

Document: "{document}"
Summary:"""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_few_shot_examples(
    examples: List[Dict],
    input_key: str,
    output_key: str,
    separator: str = "\n\n"
) -> str:
    """Format examples into a few-shot prompt string."""
    formatted = []
    for ex in examples:
        formatted.append(f"Input: {ex[input_key]}\nOutput: {ex[output_key]}")
    return separator.join(formatted)


def create_few_shot_prompt(
    examples: List[Dict],
    query: str,
    instruction: str = "",
    input_key: str = "input",
    output_key: str = "output"
) -> str:
    """Create a complete few-shot prompt with examples and query."""
    examples_str = format_few_shot_examples(examples, input_key, output_key)
    
    prompt = ""
    if instruction:
        prompt += f"{instruction}\n\n"
    prompt += f"Examples:\n{examples_str}\n\n"
    prompt += f"Input: {query}\nOutput:"
    
    return prompt


# =============================================================================
# FEW-SHOT PROMPT REGISTRY
# =============================================================================

FEW_SHOT_PROMPTS = {
    "sentiment_analysis": {
        "template": SENTIMENT_FEW_SHOT_PROMPT,
        "examples": SENTIMENT_ANALYSIS_EXAMPLES,
        "task_type": "classification"
    },
    "named_entity_recognition": {
        "template": NER_FEW_SHOT_PROMPT,
        "examples": NER_EXAMPLES,
        "task_type": "extraction"
    },
    "code_generation": {
        "template": CODE_FEW_SHOT_PROMPT,
        "examples": CODE_GENERATION_EXAMPLES,
        "task_type": "generation"
    },
    "sql_generation": {
        "template": SQL_FEW_SHOT_PROMPT,
        "examples": SQL_GENERATION_EXAMPLES,
        "task_type": "generation"
    },
    "intent_classification": {
        "template": INTENT_FEW_SHOT_PROMPT,
        "examples": INTENT_CLASSIFICATION_EXAMPLES,
        "task_type": "classification"
    },
    "summarization": {
        "template": SUMMARIZATION_FEW_SHOT_PROMPT,
        "examples": SUMMARIZATION_EXAMPLES,
        "task_type": "generation"
    }
}

