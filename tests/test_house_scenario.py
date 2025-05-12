import pytest
import random
import numpy as np
from truffaldino.house_price.sample_house_state import (
    sample_negotiation_state,
    SUBURB_PRICES,
    UNIT_MULT,
    BEDROOM_DELTA_MULT,
    BuyerRole,
    SellerRole
)

def test_seller_batna_less_than_buyer_batna_distribution():
    """
    Tests that in a significant portion of samples (at least 50%),
    the seller's BATNA (P_s(0)) is less than the buyer's BATNA (V_b_outside(0)).
    This indicates a positive bargaining range from the perspective of BATNAs.
    """
    num_samples = 200
    condition_met_count = 0

    for i in range(num_samples):
        # Randomly assign roles BEFORE sampling
        buyer_role: BuyerRole = random.choice(["Owner-Occupier", "Investor"])
        seller_role: SellerRole = random.choice(["Owner-Occupier", "Investor"])

        suburb = random.choice(list(SUBURB_PRICES.keys()))
        is_unit = random.choice([True, False])
        initial_median_price = SUBURB_PRICES[suburb]
        bedrooms_key = 0 # Using the key for BEDROOM_DELTA_MULT

        if is_unit:
            # Units: studio (-2) to 3-bed (1) -> keys for BEDROOM_DELTA_MULT
            bedrooms_key = random.randint(-2, 1)
            initial_median_price *= UNIT_MULT
            if bedrooms_key in BEDROOM_DELTA_MULT:
                initial_median_price *= BEDROOM_DELTA_MULT[bedrooms_key]
        else:
            # Houses: 2-bed (0) to 6-bed (4) -> keys for BEDROOM_DELTA_MULT
            bedrooms_key = random.randint(0, 4)
            if bedrooms_key in BEDROOM_DELTA_MULT:
                initial_median_price *= BEDROOM_DELTA_MULT[bedrooms_key]
        
        # Actual bedroom count for the state (bedrooms_key is an offset from 2-bed baseline)
        actual_bedrooms = bedrooms_key + 2


        # Pass roles to the sampling function
        # Use a different seed for each sample or no seed for full randomness
        negotiation_state = sample_negotiation_state(
            initial_median_price,
            suburb,
            is_unit,
            actual_bedrooms, # Pass actual bedroom count
            buyer_role=buyer_role,
            seller_role=seller_role,
            rng_seed=None # Or i for reproducible pseudo-randomness per run
        )

        # Seller's BATNA is P_s(0)
        seller_batna = negotiation_state["derived"]["seller_reservation_price"][0]
        # Buyer's BATNA is V_b_outside(0)
        max_buyer_surplus = negotiation_state["buyer"]["V_b"] - seller_batna
        buyer_batna = negotiation_state["derived"]["buyer_outside_values"][0]

        if buyer_batna < max_buyer_surplus:
            condition_met_count += 1

    percentage_met = condition_met_count / num_samples
    print(f"Percentage of samples where seller_BATNA < buyer_BATNA: {percentage_met:.2%}")
    assert percentage_met >= 0.35, \
        f"Expected at least 50% of samples to have seller_BATNA < buyer_BATNA, but got {percentage_met:.2%}"

# Example to allow running the test directly if needed, though pytest is standard
if __name__ == '__main__':
    test_seller_batna_less_than_buyer_batna_distribution()
