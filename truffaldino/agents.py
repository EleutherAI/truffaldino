import abc
from typing import Callable, Any, Protocol

class LLMCallable(Protocol):
    def __call__(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 512, **kwargs: Any) -> str:
        ...

class BaseAgent(abc.ABC):
    """Abstract base class for all agents."""
    def __init__(self, call_fn: LLMCallable, name: str):
        if not callable(call_fn):
            raise TypeError(f"Expected a callable for call_fn, got {type(call_fn)}")
        self.call = call_fn
        self.name = name

    @abc.abstractmethod
    def act(self, context: str) -> str:
        """Generates the agent's response based on the context."""
        pass

class PartyAgent(BaseAgent):
    """Represents a negotiating party (e.g., buyer, seller)."""
    def __init__(self, call_fn: LLMCallable, name: str, role: str):
        super().__init__(call_fn, name)
        self.role = role # e.g., 'seller', 'buyer', 'dept_A'

    def act(self, context: str) -> str:
        # In a real scenario, might add role-specific logic or prompt wrapping here
        return self.call(context)

class MediatorAgent(BaseAgent):
    """Represents the mediator agent."""
    def __init__(self, call_fn: LLMCallable, name: str = "Mediator"):
        super().__init__(call_fn, name)
        self.role = "agent"

    def act(self, context: str) -> str:
        # Mediator might have different prompt structuring logic
        return self.call(context)

# Example stub/dummy callable for testing
def dummy_llm_call(prompt: str, **kwargs: Any) -> str:
    print(f"--- DUMMY LLM CALL ({kwargs.get('agent_name', 'Unknown')}) ---")
    print(prompt)
    print("------------------------------------------------------------")
    if "party" in kwargs.get('agent_name', '').lower():
        return f"Free text response.\n\n```json\n{{\"action\": \"offer\", \"price\": {500000 + hash(prompt) % 100000}}}\n```"
    else: # Assume mediator
        return f"Mediator passing along.\n\n```json\n{{\"action\": \"offer\", \"price\": {550000 + hash(prompt) % 100000}}}\n```" 