"""
Tool Use and Function Calling Prompt Examples
Testing if AIBOM can detect tool/function calling patterns.
"""
import json
from typing import List, Dict, Any

# =============================================================================
# OPENAI FUNCTION CALLING FORMAT
# =============================================================================

OPENAI_TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search a database for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "table": {
                        "type": "string",
                        "description": "The database table to search"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Email recipient"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in a sandboxed environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# =============================================================================
# ANTHROPIC TOOL USE FORMAT
# =============================================================================

ANTHROPIC_TOOLS_DEFINITION = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g., AAPL, GOOGL"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Event title"
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in ISO format"
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in ISO format"
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee email addresses"
                }
            },
            "required": ["title", "start_time", "end_time"]
        }
    },
    {
        "name": "analyze_image",
        "description": "Analyze an image and extract information",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "URL of the image to analyze"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["objects", "text", "faces", "scene"],
                    "description": "Type of analysis to perform"
                }
            },
            "required": ["image_url"]
        }
    }
]

# =============================================================================
# BEDROCK AGENT ACTION GROUPS
# =============================================================================

BEDROCK_ACTION_GROUP_SCHEMA = {
    "openapi": "3.0.0",
    "info": {
        "title": "AI Agent Actions API",
        "version": "1.0.0",
        "description": "API for AI agent actions and tools"
    },
    "paths": {
        "/search": {
            "post": {
                "operationId": "searchKnowledgeBase",
                "summary": "Search the knowledge base for relevant information",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query"
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "description": "Number of results to return"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Search results"
                    }
                }
            }
        },
        "/execute-task": {
            "post": {
                "operationId": "executeTask",
                "summary": "Execute a predefined task",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "task_type": {
                                        "type": "string",
                                        "enum": ["summarize", "translate", "analyze"]
                                    },
                                    "input_data": {
                                        "type": "string"
                                    }
                                },
                                "required": ["task_type", "input_data"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Task result"
                    }
                }
            }
        }
    }
}

# =============================================================================
# LANGCHAIN TOOL DEFINITIONS
# =============================================================================

LANGCHAIN_TOOL_DEFINITIONS = '''
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum results to return")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information about a topic."""
    # Implementation
    pass

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().isoformat()
'''

# =============================================================================
# TOOL SELECTION PROMPT
# =============================================================================

TOOL_SELECTION_PROMPT = """You have access to the following tools:

{tools_description}

Based on the user's request, decide which tool(s) to use.

User request: {user_request}

Think about which tool is most appropriate:
1. What is the user trying to accomplish?
2. Which tool can best help with this?
3. What parameters should I pass?

Tool selection:
Tool name:
Parameters:"""

MULTI_TOOL_PROMPT = """You have access to multiple tools. Use them as needed to complete the task.

Available tools:
{tools}

Task: {task}

Plan your approach:
1. What steps are needed?
2. Which tools for each step?
3. What's the order of operations?

Execution:"""

# =============================================================================
# TOOL USE EXAMPLES
# =============================================================================

TOOL_USE_EXAMPLES = [
    {
        "user_query": "What's the weather like in Tokyo?",
        "thought": "The user wants weather information for a specific location. I should use the get_weather tool.",
        "tool_call": {
            "name": "get_weather",
            "arguments": {
                "location": "Tokyo, Japan",
                "unit": "celsius"
            }
        }
    },
    {
        "user_query": "Send an email to john@example.com about the meeting tomorrow",
        "thought": "The user wants to send an email. I need to use the send_email tool with the recipient and compose a subject and body.",
        "tool_call": {
            "name": "send_email",
            "arguments": {
                "to": "john@example.com",
                "subject": "Meeting Tomorrow",
                "body": "Hi John,\n\nThis is a reminder about our meeting tomorrow.\n\nBest regards"
            }
        }
    },
    {
        "user_query": "Calculate 15% tip on a $67.50 bill",
        "thought": "This is a calculation task. I should use the calculate tool.",
        "tool_call": {
            "name": "execute_code",
            "arguments": {
                "code": "bill = 67.50\ntip_percent = 0.15\ntip = bill * tip_percent\nprint(f'Tip: ${tip:.2f}')"
            }
        }
    }
]

# =============================================================================
# MCP (MODEL CONTEXT PROTOCOL) TOOL DEFINITIONS
# =============================================================================

MCP_TOOLS = {
    "server": "example-mcp-server",
    "tools": [
        {
            "name": "read_file",
            "description": "Read contents of a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "write_file",
            "description": "Write contents to a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "execute_command",
            "description": "Execute a shell command",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    ]
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_tools_for_prompt(tools: List[Dict]) -> str:
    """Format tool definitions for inclusion in prompts."""
    lines = []
    for tool in tools:
        if "function" in tool:  # OpenAI format
            func = tool["function"]
            lines.append(f"- {func['name']}: {func['description']}")
        elif "input_schema" in tool:  # Anthropic format
            lines.append(f"- {tool['name']}: {tool['description']}")
        else:
            lines.append(f"- {tool.get('name', 'unknown')}: {tool.get('description', '')}")
    return "\n".join(lines)


def create_tool_call_message(tool_name: str, arguments: Dict[str, Any]) -> Dict:
    """Create a tool call message in OpenAI format."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{tool_name}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments)
                }
            }
        ]
    }


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOL_REGISTRIES = {
    "openai": OPENAI_TOOLS_DEFINITION,
    "anthropic": ANTHROPIC_TOOLS_DEFINITION,
    "bedrock": BEDROCK_ACTION_GROUP_SCHEMA,
    "mcp": MCP_TOOLS,
}

