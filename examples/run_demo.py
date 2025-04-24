import sys
import os
import argparse
import pprint

# Ensure the negotiation_env package is importable
# This adjusts the path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from negotiation_env.agents import PartyAgent, MediatorAgent, dummy_llm_call
from negotiation_env.scenarios import get_scenario
from negotiation_env.core import NegotiationSession

def main():
    parser = argparse.ArgumentParser(description="Run a demo negotiation session.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="house_price",
        choices=["house_price"], # Add more choices as scenarios are implemented
        help="Name of the negotiation scenario to run."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = parser.parse_args()

    print(f"Running demo for scenario: {args.scenario} with seed: {args.seed}")

    # --- Initialize Scenario ---
    try:
        scenario = get_scenario(args.scenario)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- Initialize Agents (using dummy callable) ---
    # In a real run, replace dummy_llm_call with your actual LLM API call function
    seller_agent = PartyAgent(
        call_fn=lambda prompt, **kwargs: dummy_llm_call(prompt, agent_name="Party Seller", **kwargs),
        name="SellerBot",
        role="seller"
    )
    buyer_agent = PartyAgent(
        call_fn=lambda prompt, **kwargs: dummy_llm_call(prompt, agent_name="Party Buyer", **kwargs),
        name="BuyerBot",
        role="buyer"
    )
    mediator_agent = MediatorAgent(
        call_fn=lambda prompt, **kwargs: dummy_llm_call(prompt, agent_name="Mediator", **kwargs),
        name="MediatorBot"
    )

    # --- Create and Run Session ---
    session = NegotiationSession(
        scenario=scenario,
        party_A=seller_agent, # Assign A/B roles based on scenario if needed
        party_B=buyer_agent,
        mediator=mediator_agent
    )

    results = session.run(seed=args.seed)

    print("\n--- Final Results ---")
    pprint.pprint(results)

if __name__ == "__main__":
    main() 