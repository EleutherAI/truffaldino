import abc
from typing import Callable, Any, Protocol, List, Dict, Literal
# import inspect # No longer needed
import json

class LLMCallable(Protocol):
    # Assuming LLMCallable might also be synchronous now if agents are fully synchronous
    def __call__(self, messages: List[Dict[str, str]], *, agent_name: str, temperature: float = 0.7, max_tokens: int = 512, **kwargs: Any) -> str:
        ...

class BaseAgent(abc.ABC):
    """Abstract base class for all agents."""
    def __init__(self, call_fn: Callable, name: str, role: str):
        if not callable(call_fn):
            raise TypeError(f"Expected a callable for call_fn, got {type(call_fn)}")
        self.call_fn = call_fn
        self.name = name
        self.role = role

    @abc.abstractmethod
    def act(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generates a response from the agent based on the provided prompt and keyword arguments."""
        pass

class PartyAgent(BaseAgent):
    """Represents a negotiating party (e.g., buyer, seller)."""
    def __init__(self, call_fn: Callable, name: str, role: Literal["seller", "buyer"]):
        super().__init__(call_fn, name, role)

    def act(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return self.call_fn(messages, agent_name=self.name, **kwargs)

class MediatorAgent(BaseAgent):
    """Represents the mediator agent."""
    def __init__(self, call_fn: Callable, name: str):
        super().__init__(call_fn, name, role="mediator") # Role is fixed for Mediator

    def act(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return self.call_fn(messages, agent_name=self.name, **kwargs)

# Example stub/dummy callable for testing
# Update dummy_llm_call to accept messages List[Dict[str,str]]
def dummy_llm_call(messages: List[Dict[str,str]], **kwargs: Any) -> str:
    print(f"--- DUMMY LLM CALL ({kwargs.get('agent_name', 'Unknown')}) ---")
    # For dummy, just use the last user/system message content as the "prompt"
    prompt_content = "No message found" 
    if messages:
        last_message_content = messages[-1].get("content", "")
        if isinstance(last_message_content, str):
            prompt_content = last_message_content
        elif isinstance(last_message_content, list): # For multimodal content, just take first text part
            for item in last_message_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    prompt_content = item.get("text", "")
                    break
            
    print(f"Effective prompt for dummy: {prompt_content}")
    print("------------------------------------------------------------")
    if "party" in kwargs.get('agent_name', '').lower():
        # Parties should return JSON for action, thoughts, message as per party_prompt.txt
        return json.dumps({
            "thoughts": "This is a dummy thought from a party.",
            "action": "offer", 
            "price": 500000 + hash(prompt_content) % 100000,
            "message": f"Dummy offer based on: {prompt_content[:50]}..."
        })
    else: # Assume mediator
        # Mediator should return JSON for operation arguments as per system prompt in run_demo
        # This dummy is simplified, a real mediator would decide on operation.
        return json.dumps({
            "operation": "send_message_to_party",
            "mediator_json_payload_str": json.dumps({
                "action": "offer", 
                "price": 550000 + hash(prompt_content) % 100000,
                "recipient": "buyer", # Or seller, dummy choice
                "message": f"Dummy mediator relaying offer based on: {prompt_content[:50]}..."
            })
        }) 