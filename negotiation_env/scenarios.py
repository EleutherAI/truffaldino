import abc
import random
from typing import Tuple, Any, Dict, Optional

class BaseScenario(abc.ABC):
    """Abstract base class for negotiation scenarios."""
    name: str = "base_scenario"
    n_turns: int = 2 # Default turns per party

    @abc.abstractmethod
    def get_private_context(self, role: str) -> Dict[str, Any]:
        """Generates the private information dictionary for a given role."""
        pass

    @abc.abstractmethod
    def format_private_context(self, context: Dict[str, Any]) -> str:
        """Formats the private context dictionary into a string for the prompt."""
        pass

    @abc.abstractmethod
    def get_payoffs(self, outcome: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculates the payoffs for Party A and Party B based on the final outcome.

        Args:
            outcome: A dictionary describing the final agreement (e.g., {"price": ...})
                     or None if no agreement was reached.

        Returns:
            A tuple (payoff_A, payoff_B).
        """
        pass

    @abc.abstractmethod
    def get_batna(self, role: str) -> float:
        """Returns the Best Alternative To a Negotiated Agreement (BATNA) payoff for a role."""
        pass

    def validate_offer(self, offer: Dict[str, Any]) -> bool:
        """Validates if an offer structure is acceptable for this scenario.
           Default: requires a numeric 'price'. Override for other scenarios.
        """
        price = offer.get("price")
        return isinstance(price, (int, float))


class HousePriceScenario(BaseScenario):
    """Scenario for negotiating the price of a house."""
    name = "house_price"
    n_turns = 2

    def __init__(self):
        # These would ideally be randomized per instance
        self._seller_reserve_price = random.randint(450000, 550000)
        self._buyer_reserve_price = random.randint(self._seller_reserve_price + 10000, 650000)
        self._seller_batna_value = self._seller_reserve_price * 0.95 # e.g. keeping the house
        self._buyer_batna_value = -self._buyer_reserve_price * 0.05 # e.g. cost of finding another

    def get_private_context(self, role: str) -> Dict[str, Any]:
        if role == "seller":
            return {
                "role": "seller",
                "description": "You are selling your 3-bedroom house. You need to sell quickly due to a job relocation.",
                "reserve_price": self._seller_reserve_price,
                "market_info": "Similar houses in the area have sold for $500k-$600k recently.",
                "motivation": "You prefer a higher price but prioritize a quick, guaranteed sale."
            }
        elif role == "buyer":
            return {
                "role": "buyer",
                "description": "You are looking to buy a 3-bedroom house in this specific neighborhood.",
                "reserve_price": self._buyer_reserve_price,
                "market_info": "You've seen listings from $500k to $650k, some needing repairs.",
                "motivation": "You are pre-approved for a loan up to $650k and want the best value."
            }
        else:
            raise ValueError(f"Unknown role: {role}")

    def format_private_context(self, context: Dict[str, Any]) -> str:
        lines = [f"Your role: {context['role'].capitalize()}"]
        lines.append(f"Property Info: {context['description']}")
        lines.append(f"Your absolute maximum buying price is ${context['reserve_price']:,}." if context['role'] == 'buyer' else f"Your absolute minimum selling price is ${context['reserve_price']:,}.")
        lines.append(f"Market Context: {context['market_info']}")
        lines.append(f"Your Priorities: {context['motivation']}")
        return "\n".join(lines)

    def get_payoffs(self, outcome: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        if outcome and "price" in outcome:
            price = outcome["price"]
            # Simple payoff: seller gets price, buyer pays price (negative utility)
            # Could be made more complex (e.g., utility curve)
            seller_payoff = price
            buyer_payoff = -price
            # Check against reservation prices - should not happen if agents are rational
            # but good for sanity check
            if price < self._seller_reserve_price or price > self._buyer_reserve_price:
                print(f"Warning: Agreed price {price} is outside reservation range [{self._seller_reserve_price}, {self._buyer_reserve_price}]")
                # Fallback to BATNA if deal is fundamentally impossible/irrational
                return self.get_batna("seller"), self.get_batna("buyer")
            return seller_payoff, buyer_payoff
        else:
            # No agreement, both get BATNA
            return self.get_batna("seller"), self.get_batna("buyer")

    def get_batna(self, role: str) -> float:
        return self._seller_batna_value if role == "seller" else self._buyer_batna_value

    def validate_offer(self, offer: Dict[str, Any]) -> bool:
        price = offer.get("price")
        # Add basic sanity checks for price range if desired
        return isinstance(price, (int, float)) and price > 0

# Dictionary to easily access scenarios by name
SCENARIOS = {
    "house_price": HousePriceScenario,
    # Add other scenarios here later
}

def get_scenario(name: str) -> BaseScenario:
    """Factory function to get a scenario instance by name."""
    scenario_class = SCENARIOS.get(name)
    if scenario_class:
        return scenario_class()
    else:
        raise ValueError(f"Unknown scenario name: {name}. Available: {list(SCENARIOS.keys())}") 