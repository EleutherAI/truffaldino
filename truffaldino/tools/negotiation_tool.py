import logging
import os
import random
from typing import Any, Optional, Tuple, Dict, Literal, Union
from uuid import uuid4

from .base import BaseTool
from .schemas import OpenAIFunctionToolSchema, FunctionDefinition, FunctionParameters
from ..core import NegotiationSession # Note: Will create this path later if needed
from ..agents import PartyAgent # Assuming PartyAgent can be imported
from ..scenarios import get_scenario # Assuming get_scenario can be imported
from ..llm_call import openrouter_llm_call # Default LLM call function
from ..parse import extract_json_block, ParseError # Added ParseError

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO")) # Default to INFO for more visibility initially

DEFAULT_LLM_CONFIG = {"temperature": 0.7, "max_tokens": 1000, "model": "gpt-3.5-turbo"} # Added default model

class NegotiationTool(BaseTool):
    """A tool to manage and step through a negotiation session.

    Operations:
    - `initialize_session`: Sets up a new negotiation.
    - `send_message_to_party`: Processes a mediator's message and gets the addressed party's response.
    - `get_payoffs`: Calculates payoffs, typically at the end of a negotiation.
    - `get_subjective_rating`: Gets a subjective rating from a party post-negotiation.
    """

    def __init__(self, config: Optional[dict] = None, tool_schema: Optional[OpenAIFunctionToolSchema] = None, buyer_llm_config: Optional[dict] = None, seller_llm_config: Optional[dict] = None):
        config = config or {}
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema(
                function=FunctionDefinition(
                    name="truffaldino_mediator_step",
                    description="Manages a negotiation session. Allows initializing, sending messages between parties via a mediator, and retrieving results.",
                    parameters=FunctionParameters(
                        properties={
                            "operation": {
                                "type": "string",
                                "description": "The specific action to perform.",
                                "enum": ["initialize_session", "send_message_to_party", "get_payoffs", "get_subjective_rating"]
                            },
                            "seed": {"type": "integer", "description": "(For initialize_session, optional) Random seed for reproducibility."},
                            "mediator_json_payload_str": {"type": "string", "description": "(For send_message_to_party) The JSON string output from the mediator LLM, including 'action', 'message', and 'recipient'."},
                            "final_price": {"type": "number", "description": "(For get_payoffs, optional) The final agreed price. If not provided, uses session's outcome if available."},
                            "party_id": {"type": "string", "description": "(For send_message_to_party and get_subjective_rating) The party to speak to ('seller' or 'buyer')."}
                        },
                        required=["operation"]
                    )
                )
            )
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        self.buyer_llm_config = buyer_llm_config or {}
        self.seller_llm_config = seller_llm_config or {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        instance_id = instance_id or str(uuid4())
        if instance_id not in self._instance_dict:
            self._instance_dict[instance_id] = {
                "session": None,
                "is_initialized": False,
            }
            logger.info(f"NegotiationTool instance created: {instance_id}")
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        operation = parameters.get("operation")
        instance_data = self._instance_dict[instance_id]
        session: Optional[NegotiationSession] = instance_data.get("session")

        logger.info(f"Instance {instance_id}: Executing operation '{operation}'. Parameters: {parameters}")

        if operation == "initialize_session":
                scenario_name = "house_price"
                
                seed = parameters.get("seed") # Optional, can be None

                # Prepare LLM call functions with merged configs
                def make_llm_call_fn(specific_config: dict):
                    # Merge specific config with defaults, specific takes precedence
                    merged_config = {**DEFAULT_LLM_CONFIG, **specific_config}                    
                    return lambda prompt, **call_kwargs: openrouter_llm_call(prompt, **{**merged_config, **call_kwargs})

                seller_agent = PartyAgent(
                    call_fn=make_llm_call_fn(self.seller_llm_config),
                    name=self.seller_llm_config.get("name", "SellerBot"), # Allow name in config
                    role="seller"
                )
                buyer_agent = PartyAgent(
                    call_fn=make_llm_call_fn(self.buyer_llm_config),
                    name=self.buyer_llm_config.get("name", "BuyerBot"),
                    role="buyer"
                )

                scenario = get_scenario(scenario_name, seed=seed)
                new_session = NegotiationSession(scenario, seller_agent, buyer_agent)
                
                instance_data["session"] = new_session
                instance_data["is_initialized"] = True
                
                first_party_id, party_action_json, party_raw_response = new_session.initialize_first_turn()
                
                message_for_mediator = new_session._prepare_msg_for_mediator_view()

                logger.info(f"Instance {instance_id}: Session initialized. First party: {first_party_id}, action: {party_action_json}")
                return (
                    f"{message_for_mediator}",
                    0.0, 
                    {
                        "status": "initialized",
                        "party_id_to_speak_to_mediator": first_party_id,
                        "party_action_json": party_action_json, # Parsed action
                        "party_raw_response": party_raw_response, # Raw LLM output from party
                        # "initial_messages_for_mediator": initial_messages_for_mediator # The external mediator can use raw_response
                    }
                )
        
        if operation == "send_message_to_party":

            mediator_json_payload_str = parameters.get("mediator_json_payload_str")
            if not mediator_json_payload_str:
                return "Error: mediator_json_payload_str is required for send_message_to_party.", -1.0, {"error": "Missing mediator_json_payload_str"}
            
            try:
                mediator_action_json = extract_json_block(mediator_json_payload_str)
                if not mediator_action_json or not isinstance(mediator_action_json.get("action"), str) or not mediator_action_json.get("recipient"):
                    raise ParseError("Mediator JSON must include 'action' and 'recipient'.")
            except ParseError as pe:
                logger.warning(f"Instance {instance_id}: Parse error for mediator JSON: {pe}. Payload: {mediator_json_payload_str}")
                return f"Error: Invalid mediator JSON payload: {pe}. It must be a valid JSON string with 'action', 'message', and 'recipient' fields.", -0.5, {"error": f"Invalid mediator JSON: {pe}"}
            
            next_party_id, next_party_action, next_party_raw_resp, is_finished = session.process_mediator_message_and_get_next_party_response(
                mediator_action_json, 
                mediator_json_payload_str # Pass raw string for full logging in transcript if needed
            )

            if is_finished:
                logger.info(f"Instance {instance_id}: Negotiation finished. Results: {session.results}")
                return (
                    f"Negotiation finished. Access results using get_payoffs or from metrics.",
                    0.0, # Final reward might be calculated via calc_reward or if payoffs are positive
                    {"status": "finished", "results": session.results}
                )
            else:
                mediator_update_message = session._prepare_msg_for_mediator_view()
                return (
                    f"{mediator_update_message}",
                    0.0,
                    {
                        "status": "continue",
                        "party_id_to_speak_to_mediator": next_party_id,
                        "party_action_json": next_party_action,
                        "party_raw_response": next_party_raw_resp,
                    }
                )

        elif operation == "get_payoffs":
                if not session.results: # If negotiation didn't formally end via is_finished but we want to see current state
                    session._finalize_negotiation_results() # Attempt to finalize based on current state

                payoff_A = session.results.get("payoff_A")
                payoff_B = session.results.get("payoff_B")
                norm_payoff_A = session.score_by_payoff(session.party_A)
                norm_payoff_B = session.score_by_payoff(session.party_B)
                final_offer = session.results.get("final_offer")
                accepted = session.results.get("accepted")

                logger.info(f"Instance {instance_id}: Retrieved payoffs. Accepted: {accepted}, Final Offer: {final_offer}")
                return (
                    f"Payoffs: Seller={payoff_A:.2f} (Norm: {norm_payoff_A:.2f}), Buyer={payoff_B:.2f} (Norm: {norm_payoff_B:.2f}). Accepted: {accepted}. Offer: {final_offer}",
                    0.0, 
                    {
                        "seller_payoff": payoff_A,
                        "buyer_payoff": payoff_B,
                        "normalized_seller_payoff": norm_payoff_A,
                        "normalized_buyer_payoff": norm_payoff_B,
                        "accepted": accepted,
                        "final_offer": final_offer,
                        "full_results": session.results # Include all results for completeness
                    }
                )
        
        elif operation == "get_subjective_rating":

            party_id_str = parameters.get("party_id")
            if party_id_str not in ["seller", "buyer"]:
                return "Error: party_id must be 'seller' or 'buyer' for get_subjective_rating.", -1.0, {"error": "Invalid party_id"}
            
            party_to_rate = session.party_A if party_id_str == "seller" else session.party_B
            
            if not session.results:
                session._finalize_negotiation_results() # Attempt to finalize if not already
            
            if not session.results:
                    return "Error: No negotiation results found to get rating.", -1.0, {"error": "Results not available for rating"}

            rating = session.score_by_agent_eval(party_to_rate)
            logger.info(f"Instance {instance_id}: Subjective rating for {party_id_str}: {rating}")
            return (
                f"Subjective rating from {party_id_str.capitalize()}: {rating:.2f} (out of 1.0)",
                0.0, 
                {"party_id": party_id_str, "rating": rating}
            )

        else:
            logger.warning(f"Instance {instance_id}: Unknown operation '{operation}'.")
            return f"Error: Unknown operation '{operation}'", -1.0, {"error": f"Unknown operation: {operation}"}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        if instance_id not in self._instance_dict or not self._instance_dict[instance_id].get("session") or not self._instance_dict[instance_id]["session"].results:
            return 0.0
        
        session: NegotiationSession = self._instance_dict[instance_id]["session"]

        norm_B = session.score_by_payoff(session.party_B)

        return norm_B


    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            logger.info(f"Releasing NegotiationTool instance: {instance_id}")
            # Potentially more cleanup if sessions hold external resources not GC'd
            del self._instance_dict[instance_id]
        else:
            logger.warning(f"Attempted to release non-existent instance ID: {instance_id}") 