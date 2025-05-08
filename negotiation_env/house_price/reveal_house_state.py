import json
import random
from typing import Dict, Any, Tuple, Literal, TYPE_CHECKING
import time
import numpy as np

from negotiation_env.llm_call import openrouter_llm_call
# Import roles from sample_house_state

PROMPT_TEMPLATE_FILE = "negotiation_env/house_price/generate_context_prompt.txt"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Conditionally import only for type checkers
if TYPE_CHECKING:
    from negotiation_env.house_price.sample_house_state import BuyerRole, SellerRole

def _calculate_history_lengths(
    price_history_len: int, buyer_role: "BuyerRole", seller_role: "SellerRole"
) -> Dict[str, int]:
    """Calculates the number of historical prices to show to agent, buyer, and seller."""
    n_hist_total = price_history_len
    n_hist_agent = n_hist_total

    if buyer_role == "Investor":
        n_hist_buyer_div = 3
    else:
        n_hist_buyer_div = 10

    if seller_role == "Investor":
        n_hist_seller_div = 3
    else:
        n_hist_seller_div = 5

    # Ensure divisor is at least 1 to prevent ZeroDivisionError if n_hist_total is 0
    # Add small epsilon to random range if n_hist_total // 10 is 0
    rand_range_buyer = max(1, n_hist_total // 10)
    rand_range_seller = max(1, n_hist_total // 10)


    n_hist_buyer = max(1, n_hist_total // n_hist_buyer_div + random.randint(-rand_range_buyer, rand_range_buyer)) if n_hist_total > 0 else 0
    n_hist_seller = max(1, n_hist_total // n_hist_seller_div + random.randint(-rand_range_seller, rand_range_seller)) if n_hist_total > 0 else 0
    
    # Cap at total history length
    n_hist_buyer = min(n_hist_buyer, n_hist_total)
    n_hist_seller = min(n_hist_seller, n_hist_total)


    return {
        "agent": n_hist_agent,
        "buyer": n_hist_buyer,
        "seller": n_hist_seller,
    }

def load_prompt_template() -> str:
    """Loads the prompt template from the file."""
    try:
        with open(PROMPT_TEMPLATE_FILE, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file not found at {PROMPT_TEMPLATE_FILE}")
        raise
    except Exception as e:
        print(f"Error reading prompt template file: {e}")
        raise

def format_prompt(template: str, state: Dict[str, Any], buyer_role: "BuyerRole", seller_role: "SellerRole") -> str:
    """Formats the prompt template with data from the negotiation state."""
    # Assign history lengths (example logic, can be customized)
    history_lengths = _calculate_history_lengths(len(state['price_history']), buyer_role, seller_role)
    n_hist_agent = history_lengths["agent"]
    n_hist_buyer = history_lengths["buyer"]
    n_hist_seller = history_lengths["seller"]

    # Pre-compute unit description string
    unit_description = 'unit' if state['house_details']['unit'] else 'house'

    format_args = {
        "buyer_role": buyer_role,
        "seller_role": seller_role,
        "suburb": state['house_details']['suburb'],
        "is_unit": state['house_details']['unit'],
        "unit_description": unit_description,
        "bedrooms": state['house_details']['bedrooms'],
        "N_hist_agent": n_hist_agent,
        "sigma_comp": state['market']['sigma_annual'],
        "lambda_m": state['market']['lambda_m_weekly'],
        "lambda_b": state['market']['lambda_b_weekly'],
        "r": state['market']['r_annual'] * 100, # Percentage
        "N_hist_buyer": n_hist_buyer,
        "delta": state['buyer']['delta'],
        "C_b": state['buyer']['C_b'],
        "N_hist_seller": n_hist_seller,
        "q": state['house']['q'],
        "C_s": state['seller']['C_s'],
        "k_s": state['seller']['k_s'],
        "D": state['seller']['D'],
        "delta_s": state['seller']['delta_s'],
    }
    return template.format(**format_args)

def generate_narrative_contexts(
    negotiation_state: Dict[str, Any],
    llm_temp: float = 0.7,
    llm_max_tokens: int = 3000, # Allow ample tokens for 3x 1000 words
) -> Dict[str, str]:
    """
    Generates narrative contexts for buyer, seller, and agent using an LLM.
    Retrieves buyer_role and seller_role from the negotiation_state.

    Args:
        negotiation_state: The dictionary containing the sampled state,
                           including 'buyer_role' and 'seller_role' keys.
        llm_temp: Temperature for LLM generation.
        llm_max_tokens: Max tokens for LLM response.

    Returns:
        A dictionary containing "buyer_context", "seller_context", and "agent_context".

    Raises:
        ValueError: If unable to generate valid contexts after retries or if roles are missing.
        KeyError: If 'buyer_role' or 'seller_role' is not found in negotiation_state.
    """
    buyer_role = negotiation_state['buyer_role']
    seller_role = negotiation_state['seller_role']

    prompt_template = load_prompt_template()
    formatted_prompt = format_prompt(prompt_template, negotiation_state, buyer_role, seller_role)
    for attempt in range(MAX_RETRIES):
        print(f"Attempt {attempt + 1} of {MAX_RETRIES} to generate narrative contexts...")
        raw_response = openrouter_llm_call(
            prompt=formatted_prompt,
            agent_name="NarrativeGenerator",
            temperature=llm_temp,
            max_tokens=llm_max_tokens,
        )

        try:
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            raw_response = raw_response.strip()
            contexts = json.loads(raw_response)

            required_keys = ["buyer_context", "seller_context", "agent_context"]
            if all(k in contexts for k in required_keys) and \
               all(isinstance(contexts[k], str) for k in required_keys):
                print("Successfully generated and parsed narrative contexts.")
                return {
                    "buyer_context": contexts["buyer_context"],
                    "seller_context": contexts["seller_context"],
                    "agent_context": contexts["agent_context"],
                }
            else:
                print("Error: LLM response missing required keys or values are not strings.")
                print(f"Raw response: {raw_response}")

        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode LLM response as JSON: {e}")
            print(f"Raw response: {raw_response}")
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Raw response: {raw_response}")


        if attempt < MAX_RETRIES - 1:
            print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)

    raise ValueError("Failed to generate valid narrative contexts after multiple retries.")

def generate_full_context(
    negotiation_state: Dict[str, Any],
    llm_temp: float = 0.7,
    llm_max_tokens: int = 3000,
) -> Dict[str, str]:
    """
    Generates narrative contexts and appends sampled comparable house prices.

    Args:
        negotiation_state: The dictionary containing the sampled state.
        llm_temp: Temperature for LLM generation.
        llm_max_tokens: Max tokens for LLM response.

    Returns:
        A dictionary containing "buyer_context", "seller_context",
        and "agent_context", with price history appended.
    """
    # Step 1: Generate base narrative contexts
    narrative_contexts = generate_narrative_contexts(
        negotiation_state, llm_temp, llm_max_tokens
    )

    # Step 2: Get roles and price history
    buyer_role = negotiation_state['buyer_role']
    seller_role = negotiation_state['seller_role']
    price_history = negotiation_state.get('price_history', [])
    price_history_len = len(price_history)

    if not price_history_len:
        return narrative_contexts

    # Step 3: Calculate history lengths
    history_lengths = _calculate_history_lengths(price_history_len, buyer_role, seller_role)

    # Step 4: Append prices to contexts
    roles_config = {
        "buyer": {
            "key_context": "buyer_context",
            "num_prices": history_lengths["buyer"],
            "append_format": "\n\nThe comparable sales prices you were provided are: [{}].",
        },
        "seller": {
            "key_context": "seller_context",
            "num_prices": history_lengths["seller"],
            "append_format": "\n\nThe comparable sales prices you were provided are: [{}].",
        },
        "agent": {
            "key_context": "agent_context",
            "num_prices": history_lengths["agent"],
            "append_format": "\n\nThe {} comparable sales data points for your analysis are: [{}].",
        },
    }

    for role, config in roles_config.items():
        context_key = config["key_context"]
        num_prices_to_sample = config["num_prices"]

        if num_prices_to_sample == 0:
            selected_prices_str = "None available"
        else:
            # Ensure sampling k is not greater than population size
            actual_num_to_sample = min(num_prices_to_sample, price_history_len)
            if actual_num_to_sample > 0:
                selected_prices = np.random.choice(price_history, size=actual_num_to_sample, replace=False)
                price_str_list = [f"${p:,.0f}" for p in selected_prices]
                selected_prices_str = ", ".join(price_str_list)
            else: # Should not happen if num_prices_to_sample > 0 and price_history_len > 0
                selected_prices_str = "None available due to sampling constraints"


        if role == "agent":
            appendix = config["append_format"].format(history_lengths["agent"], selected_prices_str)
        else:
            appendix = config["append_format"].format(selected_prices_str)
        
        narrative_contexts[context_key] += appendix

    return narrative_contexts


# Example usage (can be run standalone for testing)
if __name__ == "__main__":

    # Create a mock negotiation state for testing
    mock_state = {
        'market': {'g_annual': 0.03, 'sigma_annual': 0.08, 'r_annual': 0.05, 'lambda_m_weekly': 1.0, 'lambda_b_weekly': 0.8, 'g_daily': 8e-05, 'sigma_daily': 0.004, 'r_daily': 0.00013, 'lambda_m_daily': 0.14, 'lambda_b_daily': 0.11},
        'price_history': [900000.0, 901000.0, 900500.0] * 30, # Example history (90 points)
        'P_today': 915000.0,
        'house': {'q': 15000.0, 'V_h': 930000.0},
        'seller': {'C_s': 200.0, 'k_s': 8000.0, 'D': 45},
        'buyer': {'delta': 25000.0, 'C_b': 150.0, 'V_b': 940000.0},
        'house_details': {'suburb': 'Fitzroy, Melbourne, Australia', 'unit': False, 'bedrooms': 3},
        'derived': {'seller_option_premium': 5000.0, 'seller_reservation_price': [935200.0, 935400.0, 935600.0, 935800.0], 'buyer_outside_values': [910000.0, 909000.0, 908000.0, 907000.0]},
        'buyer_role': "Owner-Occupier", # Add role to state
        'seller_role': "Investor" # Add role to state
    }

    try:
        # Call generate_narrative_contexts with only the state
        generated_contexts = generate_narrative_contexts(mock_state)
        print("\n--- Generated Narrative Contexts (without explicit prices) ---")
        print("\nBuyer Context:\n", generated_contexts["buyer_context"])
        print("\nSeller Context:\n", generated_contexts["seller_context"])
        print("\nAgent Context:\n", generated_contexts["agent_context"])

        # Call generate_full_context to include prices
        full_generated_contexts = generate_full_context(mock_state)
        print("\n\n--- Generated Full Contexts (with explicit prices) ---")
        print("\nBuyer Context (full):\n", full_generated_contexts["buyer_context"])
        print("\nSeller Context (full):\n", full_generated_contexts["seller_context"])
        print("\nAgent Context (full):\n", full_generated_contexts["agent_context"])

    except ValueError as e:
        print(f"\nFailed to generate contexts: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
