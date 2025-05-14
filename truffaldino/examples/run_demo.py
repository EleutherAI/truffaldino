import sys
import os
import argparse
import pprint
import requests
import json
from dotenv import load_dotenv
from datetime import datetime

# Ensure the negotiation_env package is importable
# This adjusts the path to include the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from truffaldino.agents import PartyAgent, MediatorAgent
from truffaldino.scenarios import get_scenario
from truffaldino.core import NegotiationSession
from truffaldino.llm_call import openrouter_llm_call

# --- Load API Key ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env file.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run a demo negotiation session.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="house_price",
        choices=["house_price"], # Add more choices as scenarios are implemented
        help="Name of the negotiation scenario to run."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # --- Setup Logging --- 
    log_dir_base = os.path.join(project_root, "logs")
    md_log_dir = os.path.join(log_dir_base, "md")
    json_log_dir = os.path.join(log_dir_base, "json")
    os.makedirs(md_log_dir, exist_ok=True)
    os.makedirs(json_log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
    log_filename_base = f"{args.scenario}_{timestamp}{seed_suffix}"
    md_log_path = os.path.join(md_log_dir, f"{log_filename_base}.md")
    json_log_path = os.path.join(json_log_dir, f"{log_filename_base}.json")

    print(f"Running demo for scenario: {args.scenario} with seed: {args.seed}")
    print(f"Markdown log: {md_log_path}")
    print(f"JSON log: {json_log_path}")

    # --- Initialize Scenario ---
    try:
        scenario = get_scenario(args.scenario, seed=args.seed)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- Initialize Agents (using OpenRouter callable) ---
    # Set default LLM parameters here if needed
    llm_params = {"temperature": 0.7, "max_tokens": 500} # Example

    seller_agent = PartyAgent(
        call_fn=lambda prompt, **kwargs: openrouter_llm_call(prompt, agent_name="Party Seller", **{**llm_params, **kwargs}),
        name="SellerBot (Gemini-Flash)",
        role="seller"
    )
    buyer_agent = PartyAgent(
        call_fn=lambda prompt, **kwargs: openrouter_llm_call(prompt, agent_name="Party Buyer", **{**llm_params, **kwargs}),
        name="BuyerBot (Gemini-Flash)",
        role="buyer"
    )
    mediator_agent = MediatorAgent(
        call_fn=lambda prompt, **kwargs: openrouter_llm_call(prompt, agent_name="Mediator", **{**llm_params, **kwargs}),
        name="MediatorBot (Gemini-Flash)"
    )

    # --- Create and Run Session ---
    session = NegotiationSession(
        scenario=scenario,
        party_A=seller_agent, # Assign A/B roles based on scenario if needed
        party_B=buyer_agent,
        mediator=mediator_agent
    )

    results = session.run(seed=args.seed, md_log_path=md_log_path, json_log_path=json_log_path)

    print("\n--- Final Results ---")
    pprint.pprint(results)
    return results

if __name__ == "__main__":
    main() 