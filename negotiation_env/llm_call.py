
from dotenv import load_dotenv
import requests
import json
import os

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def openrouter_llm_call(prompt: str, agent_name: str = "Unknown", temperature: float = 0.7, max_tokens: int = 512, **kwargs) -> str:
    """Calls the OpenRouter API for LLM generation."""
    print(f"--- Calling OpenRouter ({agent_name}) ---")
    # print(f"Prompt:\n{prompt}") # Optional: uncomment to see full prompts
    print("--------------------------------------")

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            json={
                "model": "google/gemini-2.5-flash-preview",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
        )
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"--- Received Response ({agent_name}) ---")
        # print(f"Response:\n{content}") # Optional: uncomment to see full responses
        print("--------------------------------------")
        return content.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter API: {e}")
        if response and "429" in response.text:
            print("Rate limit likely exceeded. Returning default reject.")
        return json.dumps({"action": "reject", "reason": "API Error"})
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing OpenRouter response: {e}")
        raw_response_text = response.text if 'response' in locals() and hasattr(response, 'text') else "No response text available"
        print(f"Raw response: {raw_response_text}")
        return json.dumps({"action": "reject", "reason": "API Response Parse/Format Error"})