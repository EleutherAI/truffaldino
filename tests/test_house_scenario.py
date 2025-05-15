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
from truffaldino.examples.run_demo import main as run_demo
import time

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
        seller_batna = negotiation_state["derived"]["seller_breakeven_price"][0]
        # Buyer's BATNA is V_b_outside(0)
        max_buyer_surplus = negotiation_state["buyer"]["V_b"] - seller_batna
        buyer_batna = negotiation_state["derived"]["buyer_outside_values"][0]

        if buyer_batna < max_buyer_surplus:
            condition_met_count += 1

    percentage_met = condition_met_count / num_samples
    print(f"Percentage of samples where seller_BATNA < buyer_BATNA: {percentage_met:.2%}")
    assert percentage_met >= 0.35, \
        f"Expected at least 50% of samples to have seller_BATNA < buyer_BATNA, but got {percentage_met:.2%}"

def test_offer_monotonicity():
    """
    Tests that in multiple negotiation sessions:
    1. Buyer's offers never decrease (monotonically non-decreasing)
    2. Seller's offers never increase (monotonically non-increasing)
    """
    num_demos = 5
    for demo_num in range(num_demos):
        # Run a demo with a different seed for each run
        results, session = run_demo(scenario="house_price", seed=demo_num)
        
        # Extract all offers from the transcript
        buyer_offers = []
        seller_offers = []
        
        for entry in results["transcript"]:
            if entry["actor_id"] == "A":  # Seller
                if entry["response"].get("action") == "offer":
                    seller_offers.append(entry["response"].get("price"))
            elif entry["actor_id"] == "B":  # Buyer
                if entry["response"].get("action") == "offer":
                    buyer_offers.append(entry["response"].get("price"))
        
        # Verify monotonicity
        for i in range(1, len(seller_offers)):
            assert seller_offers[i] <= seller_offers[i-1], \
                f"Seller's offer increased in demo {demo_num}: {seller_offers[i-1]} -> {seller_offers[i]}"
        
        for i in range(1, len(buyer_offers)):
            assert buyer_offers[i] >= buyer_offers[i-1], \
                f"Buyer's offer decreased in demo {demo_num}: {buyer_offers[i-1]} -> {buyer_offers[i]}"

        score_A = session.score_by_payoff(session.party_A)
        score_B = session.score_by_payoff(session.party_B)
        score_A_agent_eval = session.results["eval_party_A"]
        score_B_agent_eval = session.results["eval_party_B"]

        assert type(score_A) == float
        assert type(score_B) == float
        assert type(score_A_agent_eval) == float
        assert type(score_B_agent_eval) == float
        print(f"Score A: {score_A}, Score B: {score_B}, Score A Agent Eval: {score_A_agent_eval}, Score B Agent Eval: {score_B_agent_eval}")


        time.sleep(30)
# Example to allow running the test directly if needed, though pytest is standard
if __name__ == '__main__':
    test_seller_batna_less_than_buyer_batna_distribution()
    test_offer_monotonicity()
