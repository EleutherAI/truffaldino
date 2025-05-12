import abc
import numpy as np
from typing import Tuple, Any, Dict, Optional

# For HousePriceScenario
import random as py_random # Alias to avoid conflict if self.random is used
from truffaldino.house_price.sample_house_state import (
    sample_negotiation_state,
    SUBURB_PRICES, UNIT_MULT, BEDROOM_DELTA_MULT,
    BuyerRole, SellerRole # For type hinting
)
from truffaldino.house_price.reveal_house_state import generate_full_context


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
    """Scenario for negotiating the price of a house, using sampled state and LLM-generated contexts."""
    name = "house_price"
    n_turns = 3 # Can be adjusted or made dynamic based on sampled state if needed

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed
        self.rng = py_random.Random(seed)
        # np_rng is created inside sample_negotiation_state from the seed

        # 1. Determine roles
        # Explicitly type for clarity, though random.choice will pick from the Literal values
        self.buyer_role: BuyerRole = self.rng.choice(["Owner-Occupier", "Investor"])
        self.seller_role: SellerRole = self.rng.choice(["Owner-Occupier", "Investor"])

        # 2. Determine house features and P0 (initial median price for simulation)
        suburb = self.rng.choice(list(SUBURB_PRICES.keys()))
        is_unit = self.rng.choice([True, False])
        initial_median_price = SUBURB_PRICES[suburb]
        
        bedrooms_delta_key: int
        if is_unit:
            bedrooms_delta_key = self.rng.randint(-2, 1) # Studio (-2) to 3-bed (1)
            initial_median_price *= UNIT_MULT
            if bedrooms_delta_key in BEDROOM_DELTA_MULT:
                 initial_median_price *= BEDROOM_DELTA_MULT[bedrooms_delta_key]
        else: # House
            bedrooms_delta_key = self.rng.randint(0, 4) # 2-bed (0) to 6-bed (4)
            if bedrooms_delta_key in BEDROOM_DELTA_MULT:
                initial_median_price *= BEDROOM_DELTA_MULT[bedrooms_delta_key]
        
        actual_bedrooms = bedrooms_delta_key + 2 # E.g., delta key 0 means 2 bedrooms

        # 3. Call sample_negotiation_state
        # This function creates its own np.random.default_rng(rng_seed)
        self.negotiation_state = sample_negotiation_state(
            P0=initial_median_price,
            suburb=suburb,
            is_unit=is_unit,
            bedrooms=actual_bedrooms,
            buyer_role=self.buyer_role,
            seller_role=self.seller_role,
            rng_seed=self.seed 
        )

        # 4. CRITICAL: Populate/overwrite 'house_details' in the state for generate_full_context.
        # sample_negotiation_state populates a 'house_details' but it might not be complete
        # or exactly what generate_full_context expects if it relies on the __main__ block's version.
        # The __main__ block in sample_house_state.py finalizes this part.
        self.negotiation_state["house_details"] = {
            "suburb": suburb,
            "unit": is_unit,
            "bedrooms": actual_bedrooms, 
        }
        # Ensure buyer_role and seller_role are in negotiation_state (sample_negotiation_state does this)
        # self.negotiation_state['buyer_role'] = self.buyer_role
        # self.negotiation_state['seller_role'] = self.seller_role

        # 5. Generate narrative contexts using the fully prepared negotiation_state
        # generate_full_context expects 'buyer_role' and 'seller_role' keys in negotiation_state.
        self.narrative_contexts = generate_full_context(self.negotiation_state)

    def get_private_context(self, role: str) -> Dict[str, Any]:
        """Returns a dictionary containing the pre-generated narrative for the role."""
        if role == "seller":
            return {"narrative": self.narrative_contexts["seller_context"]}
        elif role == "buyer":
            return {"narrative": self.narrative_contexts["buyer_context"]}
        elif role == "mediator": # Role used by NegotiationSession to get mediator's context
            return {"narrative": self.narrative_contexts["agent_context"]}
        else:
            raise ValueError(f"Unknown role for private context: {role}. Expected 'seller', 'buyer', or 'mediator'.")

    def format_private_context(self, context: Dict[str, Any]) -> str:
        """Returns the narrative string from the context dictionary."""
        narrative = context.get("narrative")
        if narrative is None:
            raise ValueError("Context dictionary for formatting is missing 'narrative' key.")
        return str(narrative)

    def get_payoffs(self, outcome: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Calculates payoffs based on the negotiation outcome and the sampled state.
        Seller payoff = price achieved.
        Buyer payoff = Buyer's valuation (V_b) - price paid.
        If no deal, or irrational deal, they get their BATNA values.
        """
        seller_batna = self.get_batna("seller")
        buyer_batna = self.get_batna("buyer")

        if outcome and "price" in outcome and isinstance(outcome["price"], (int, float)):
            price = float(outcome["price"])
            
            # True values from the perspective of the simulation
            seller_reservation_price = float(self.negotiation_state['derived']['seller_reservation_price'][0])
            buyer_valuation = float(self.negotiation_state['buyer']['V_b'])

            # A rational deal should be within these bounds
            if price >= seller_reservation_price and price <= buyer_valuation:
                seller_payoff = price
                buyer_payoff = buyer_valuation - price
                return seller_payoff, buyer_payoff
            else:
                # Deal occurred but was outside the rational ZOPA defined by P_s(0) and V_b
                print(f"Warning: Agreed price {price} is outside rational ZOPA "
                      f"[{seller_reservation_price:.2f}, {buyer_valuation:.2f}]. Payoffs are BATNA values.")
                return seller_batna, buyer_batna
        else:
            # No agreement or invalid outcome structure
            return seller_batna, buyer_batna

    def get_batna(self, role: str) -> float:
        """
        Returns the Best Alternative To a Negotiated Agreement (BATNA) payoff.
        For the seller, this is their reservation price P_s(0).
        For the buyer, this is their outside option value V_b_outside(0).
        """
        if role == "seller":
            return float(self.negotiation_state['derived']['seller_reservation_price'][0])
        elif role == "buyer":
            return float(self.negotiation_state['derived']['buyer_outside_values'][0])
        else:
            raise ValueError(f"Unknown role for BATNA: {role}")

    def validate_offer(self, offer: Dict[str, Any]) -> bool:
        price = offer.get("price")
        # Basic validation: price must be a positive number. More complex validation can be added.
        return isinstance(price, (int, float)) and price > 0

# Dictionary to easily access scenarios by name
SCENARIOS = {
    "house_price": HousePriceScenario,
    # Add other scenarios here later
}

def get_scenario(name: str, seed: Optional[int] = None) -> BaseScenario:
    """Factory function to get a scenario instance by name, passing a seed."""
    scenario_class = SCENARIOS.get(name)
    if scenario_class:
        # Pass the seed to the scenario's constructor
        return scenario_class(seed=seed)
    else:
        raise ValueError(f"Unknown scenario name: {name}. Available: {list(SCENARIOS.keys())}") 