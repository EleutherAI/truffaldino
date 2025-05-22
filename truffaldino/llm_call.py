from openai import OpenAI
from dotenv import load_dotenv
import json
import os
from typing import List, Dict, Any

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def openrouter_llm_call(messages: List[Dict[str, str]], agent_name: str = "Unknown", temperature: float = 0.7, max_tokens: int = 1000, tools: List[Dict] = None, **kwargs: Any) -> str:
    """Calls the OpenRouter API for LLM generation using a list of messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        agent_name: Name of the agent making the call (for logging)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tools: Optional list of tool definitions in OpenAI format
        **kwargs: Additional parameters to pass to the API call
    """
    print(f"--- Calling OpenRouter ({agent_name}) ---")
    # print(f"Messages:\n{json.dumps(messages, indent=2)}") # Optional: uncomment to see full message list
    print("--------------------------------------")

    # Extract model from kwargs or use a default
    model = kwargs.pop("model", "qwen/qwen3-14b")

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        # Prepare the completion request
        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # Add tools if provided
        if tools:
            completion_params["tools"] = tools

        # Make the API call
        response = client.chat.completions.create(**completion_params)
        
        # Extract the response content
        message = response.choices[0].message
        
        return message

    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429:
            print("Rate limit likely exceeded. Returning default reject.")
        return json.dumps({"action": "reject", "reason": "API Error"})
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing OpenRouter response: {e}")
        raw_response_text = response.text if 'response' in locals() and hasattr(response, 'text') else "No response text available"
        print(f"Raw response: {raw_response_text}")
        return json.dumps({"action": "reject", "reason": "API Response Parse/Format Error"})