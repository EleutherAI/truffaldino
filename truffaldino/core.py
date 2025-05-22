import random
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Literal, Union
from io import TextIOWrapper

from transformers import AutoTokenizer

from .agents import PartyAgent, MediatorAgent, BaseAgent
from .scenarios import BaseScenario
from .parse import extract_json_block, ParseError

# Simple template loading/rendering (replace with Jinja2 for more power if needed)
TEMPLATE_DIR = Path(__file__).parent / "templates"

def load_template(name: str) -> str:
    try:
        with open(TEMPLATE_DIR / name, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template {name} not found in {TEMPLATE_DIR}")

def render_template(template_str: str, context: Dict[str, Any]) -> str:
    # Basic {{key}} replacement
    for key, value in context.items():
        template_str = template_str.replace(f"{{{{{key}}}}}", str(value))
    # Handle potential default filters like {{ current_offer_price | default('None') }}
    import re
    template_str = re.sub(r"{{\s*(\w+)\s*\|\s*default\('(.*?)'\)\s*}}", lambda m: str(context.get(m.group(1), m.group(2))), template_str)
    return template_str

class NegotiationSession:
    """Manages the state and execution of a single negotiation round."""

    def __init__(self, scenario: BaseScenario, party_A: PartyAgent, party_B: PartyAgent, tokenizer_name_or_path: str = "gpt2"):
        self.scenario = scenario
        self.party_A = party_A
        self.party_B = party_B

        self.transcript: List[Dict[str, Any]] = []
        self.turn: int = 0
        self._mediator_role = "assistant"
        self.last_turn: Dict[str, int] = {"seller": -1, "buyer": -1, self._mediator_role: -1}
        
        self.current_offer: Optional[Dict[str, Any]] = None
        self.offer_proposed_by: Optional[Literal["seller", "buyer"]] = None
        self.seller_accepted: bool = False
        self.buyer_accepted: bool = False
        
        self.current_party_id_to_act: Optional[Literal["seller", "buyer"]] = None
        self.last_party_action_json: Optional[Dict[str, Any]] = None
        self.last_mediator_message_to_party: Dict[Literal["seller", "buyer"], str] = {"seller": "", "buyer": ""}
        
        self.results: Optional[Dict[str, Any]] = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.chat_template = (
            "{% for message in messages %}"
                "{% if message.role == 'system' %}"
                    "{{ message.content \n}}"
                "{% elif message.role == 'seller' %}"
                    "\n\n**Party A (Seller) to Agent:**\n{{ message.content }}"
                "{% elif message.role == 'buyer' %}"
                    "\n\n**Party B (Buyer) to Agent:**\n{{ message.content }}"
                "{% elif message.role == 'agent' %}"
                    "\n\n**Agent:**\n{{ message.content }}"
                "{% else %}"
                    "\n\n**{{ message.role }}:**\n{{ message.content }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{% if messages[-1].role == 'buyer' or messages[-1].role == 'seller' %}"
                    "\n\n**Agent:**\n"
                 "{% endif %}"
            "{% endif %}"
        )

    def _get_party(self, party_id: Literal["seller", "buyer"]) -> PartyAgent:
        return self.party_A if party_id == "seller" else self.party_B

    def _prepare_chat_for_party_view(self, party_id: Literal["seller", "buyer"]) -> List[Dict[str, str]]:
        chat_messages = []
        for entry in self.transcript:

            # Party's messages to mediator - parties see their own thoughts
            if entry['actor_id'] == party_id and entry['recipient'] is None:
                formatted_content = []
                if 'thoughts' in entry['response'] and entry['response']['thoughts']:
                    formatted_content.append(f"  My Thoughts: {entry['response'].get('thoughts', '')}")
                formatted_content.append(f"  My Message to Agent: {entry['response'].get('message', '')}")
                formatted_content.append(f"  My Proposed Action: {entry['response'].get('action', '')}")
                if 'price' in entry['response'] and entry['response'].get('price') is not None:
                    formatted_content.append(f"  My Proposed Price: {entry['response'].get('price', '')}")
                if 'turn' in entry:
                    formatted_content.append(f"  Turn: {entry['turn']}")
                chat_messages.append({"role": entry['role'], "content": "\n".join(formatted_content)})
            # Mediator's messages to this party - parties don't see thoughts
            elif entry['role'] == self._mediator_role and entry['recipient'] == party_id:
                formatted_content = []
                formatted_content.append(f"  Message from Mediator: {entry['response'].get('message', entry['raw_response'])}")
                formatted_content.append(f"  Proposed Action: {entry['response'].get('action', '')}")
                if 'price' in entry['response'] and entry['response'].get('price') is not None:
                    formatted_content.append(f"  Proposed Price: {entry['response'].get('price', '')}")
                if 'turn' in entry:
                    formatted_content.append(f"  Turn: {entry['turn']}")    
                chat_messages.append({"role": entry['role'], "content": "\n".join(formatted_content)})
        
        return [{
            "role": "system",
            "content": self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False
            )
        }] if chat_messages else [{"role": "system", "content": "No messages yet."}]

    def _prepare_msg_for_mediator_view(self) -> List[Dict[str, str]]:
        chat_messages = []
        start_turn_val = self.last_turn[self._mediator_role] + 1
        processed_turns_in_this_call = set()

        if start_turn_val == 0:
            chat_messages.append({"role": "system", "content": self._build_mediator_system_prompt()})

        for entry in self.transcript:
            if entry['turn'] < start_turn_val:
                continue
            processed_turns_in_this_call.add(entry['turn'])

            # Messages from parties to the mediator - mediator does not see party's thoughts
            if entry['actor_id'] in ["seller", "buyer"] and entry['recipient'] is None:
                formatted_content = []
                formatted_content.append(f"  Message from {entry['role'].capitalize()} ({entry['actor_id']}): {entry['response'].get('message', '')}")
                formatted_content.append(f"  Proposed Action: {entry['response'].get('action', '')}")
                if 'price' in entry['response'] and entry['response'].get('price') is not None:
                    formatted_content.append(f"  Proposed Price: {entry['response'].get('price', '')}")
                if 'turn' in entry:
                    formatted_content.append(f"  Turn: {entry['turn']}")
                chat_messages.append({"role": entry['role'], "content": "\n".join(formatted_content)})
            # Mediator's  messages (to parties) - mediator sees own raw responses
            elif entry['role'] == self._mediator_role and entry['recipient'] is not None:

                chat_messages.append({
                    "role": self._mediator_role,
                    "content": entry['raw_response'],
                    "recipient": entry['recipient'],
                })

        if processed_turns_in_this_call:
            self.last_turn[self._mediator_role] = max(processed_turns_in_this_call)

        chat_messages = chat_messages if chat_messages else [{"role": "system", "content": "No messages yet."}]

        return_msg = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False
        )

        return return_msg

    def _build_party_prompt(self, party_id: Literal["seller", "buyer"]) -> str:
        party = self._get_party(party_id)
        private_context_dict = self.scenario.get_private_context(party.role)
        private_context_str = self.scenario.format_private_context(private_context_dict)

        base_prompt_template_str = load_template("party_prompt.txt")
        base_prompt_context = {
            "role": party.role,
            "scenario_name": self.scenario.name,
            "private_context": private_context_str,
            "current_offer_price": self.current_offer.get('price') if self.current_offer else None,
            "current_offer_by": self.offer_proposed_by,
        }
        system_prompt_content = render_template(base_prompt_template_str, base_prompt_context)

        messages = [
            {"role": "system", "content": system_prompt_content}
        ]
        
        chat_history = self._prepare_chat_for_party_view(party_id)
        messages.extend(chat_history)
        
        return messages

    def _build_mediator_system_prompt(self) -> str:
        mediator_prompt_template_str = load_template("mediator_prompt.txt")
        private_context_dict = self.scenario.get_private_context("agent")
        private_context_str = self.scenario.format_private_context(private_context_dict)

        mediator_prompt_context = {
            "role": self._mediator_role,
            "scenario_name": self.scenario.name,
            "party_A_name": self.party_A.name,
            "party_B_name": self.party_B.name,
            "private_context": private_context_str
        }
        system_prompt_content = render_template(mediator_prompt_template_str, mediator_prompt_context)
        return system_prompt_content

    def _record_turn(
            self,
            actor_role: str,
            prompt_or_payload: Union[str, Dict[str, Any]],
            raw_response: str,
            parsed_response_json: Optional[Dict[str, Any]],
            recipient_party_id: Optional[Literal["seller", "buyer"]] = None,
            actor_identifier: Optional[str] = None
            ):
        
        actor_name = actor_identifier
        if actor_role == "seller":
            actor_name = self.party_A.name
            actor_identifier = "seller"
        elif actor_role == "buyer":
            actor_name = self.party_B.name
            actor_identifier = "buyer"

        self.transcript.append({
            "turn": self.turn,
            "actor_name": actor_name,
            "actor_id": actor_identifier,
            "role": actor_role,
            "prompt_or_payload": prompt_or_payload,
            "response": parsed_response_json or {},
            "raw_response": raw_response,
            "recipient": recipient_party_id, 
        })

    def _update_state_from_action(self, action: Dict[str, Any], proposing_party_id: Literal["seller", "buyer"]):
        action_type = action.get("action")

        if action_type == "accept":
            if proposing_party_id == "seller":
                self.seller_accepted = True
            else:
                self.buyer_accepted = True
            if not self.current_offer or self.offer_proposed_by == proposing_party_id:
                 print(f"Warning: Party {proposing_party_id} accepted but no valid offer from other party exists. Treating as reject.")
                 self.seller_accepted = False
                 self.buyer_accepted = False

        elif action_type == "offer":
            if self.scenario.validate_offer(action):
                self.current_offer = action 
                self.offer_proposed_by = proposing_party_id
                self.seller_accepted = False
                self.buyer_accepted = False
            else:
                print(f"Warning: Invalid offer structure from {proposing_party_id}: {action}")
        
        elif action_type == "counteroffer":
            self.seller_accepted = False
            self.buyer_accepted = False
            if self.scenario.validate_offer(action):
                self.current_offer = action
                self.offer_proposed_by = proposing_party_id
            else:
                print(f"Warning: Invalid counter-offer structure from {proposing_party_id}: {action}")

        elif action_type == 'reject':
            self.seller_accepted = False
            self.buyer_accepted = False
        
        elif action_type == 'inquiry' or action_type == 'pass' or action_type == 'response':
            pass
        else:
            print(f"Warning: Unknown action type '{action_type}' from {proposing_party_id} in _update_state_from_action. Action: {action}")

    def _is_finished(self) -> bool:
        agreement_reached = False
        if self.current_offer is not None:
            if self.offer_proposed_by == "seller" and self.buyer_accepted:
                agreement_reached = True
            elif self.offer_proposed_by == "buyer" and self.seller_accepted:
                agreement_reached = True
        if agreement_reached:
            return True
        return self.turn >= self.scenario.n_turns * 2

    def initialize_first_turn(self) -> Tuple[Literal["seller", "buyer"], Dict[str, Any], str]:
        self.transcript = []
        self.turn = 0 
        self.current_offer = None
        self.offer_proposed_by = None
        self.seller_accepted = False
        self.buyer_accepted = False
        self.last_mediator_message_to_party = {"seller": "", "buyer": ""}
        self.results = None
        self.last_turn = {"seller": -1, "buyer": -1, self._mediator_role: -1}

        first_party_id: Literal["seller", "buyer"] = random.choice(["seller", "buyer"])
        self.current_party_id_to_act = first_party_id
        
        party_agent = self._get_party(first_party_id)
        party_prompt_messages = self._build_party_prompt(first_party_id)
        
        try:
            party_raw_response = party_agent.act(party_prompt_messages).content
            party_action_json = extract_json_block(party_raw_response)
            if not party_action_json or not isinstance(party_action_json.get("action"), str):
                print(f"Warning: Missing or invalid JSON from {first_party_id} on first turn. Raw: {party_raw_response}")
                party_action_json = {"action": "inquiry", "message": party_raw_response, "thoughts": "Initial message, JSON parsing failed."}
        except Exception as e:
            print(f"Error getting/parsing action from {party_agent.name} on first turn: {e}")
            party_action_json = {"action": "inquiry", "message": f"Error: {e}", "thoughts": "Error during initial action."}
            party_raw_response = json.dumps(party_action_json)

        self.last_party_action_json = party_action_json
        
        self._record_turn(
            actor_role=first_party_id,
            prompt_or_payload=party_prompt_messages,
            raw_response=party_raw_response,
            parsed_response_json=party_action_json,
            recipient_party_id=None,
            actor_identifier=first_party_id 
        )
        
        print(f"--- Initialization: {self.scenario.name} ---")
        print(f"Party A ({self.party_A.role}): {self.party_A.name}")
        print(f"Party B ({self.party_B.role}): {self.party_B.name}")
        print(f"First to speak (to mediator): {first_party_id}")
        print(f"Max turns per party (before external mediator check): {self.scenario.n_turns}")

        return first_party_id, party_action_json, party_raw_response

    def process_mediator_message_and_get_next_party_response(
        self, 
        mediator_action_json: Dict[str, Any], 
        mediator_raw_response: str
    ) -> Tuple[Optional[Literal["seller", "buyer"]], Optional[Dict[str, Any]], Optional[str], bool]:
        if not self.current_party_id_to_act or not self.last_party_action_json:
            raise ValueError("process_mediator_message called before a party has acted or action recorded.")

        self._update_state_from_action(self.last_party_action_json, self.current_party_id_to_act)

        recipient_party_id = mediator_action_json.get("recipient")
        if recipient_party_id not in ["seller", "buyer"]:
            print(f"Warning: Mediator JSON missing valid recipient. Got: {recipient_party_id}. Defaulting based on alternation.")
            recipient_party_id = "buyer" if self.current_party_id_to_act == "seller" else "seller"
            mediator_action_json["recipient"] = recipient_party_id

        self._record_turn(
            actor_role="mediator",
            prompt_or_payload=mediator_action_json,
            raw_response=mediator_raw_response,
            parsed_response_json=mediator_action_json,
            recipient_party_id=recipient_party_id,
            actor_identifier="external_mediator"
        )
        
        self.last_mediator_message_to_party[recipient_party_id] = mediator_action_json.get("message", mediator_raw_response)
        
        if self._is_finished() or self.turn >= self.scenario.n_turns * 2:
            self._finalize_negotiation_results()
            print("\n--- Negotiation Finished (after mediator message) ---")
            return None, None, None, True 

        self.turn += 1

        next_acting_party_id = recipient_party_id
        self.current_party_id_to_act = next_acting_party_id
        
        party_agent = self._get_party(next_acting_party_id)
        party_prompt_messages = self._build_party_prompt(next_acting_party_id)
        
        try:
            party_raw_response = party_agent.act(party_prompt_messages).content
            party_action_json = extract_json_block(party_raw_response)
            if not party_action_json or not isinstance(party_action_json.get("action"), str):
                print(f"Warning: Missing or invalid JSON from {next_acting_party_id}. Raw: {party_raw_response}")
                party_action_json = {"action": "inquiry", "message": party_raw_response, "thoughts": "JSON parsing failed."}
        except Exception as e:
            print(f"Error getting/parsing action from {party_agent.name}: {e}")
            party_action_json = {"action": "inquiry", "message": f"Party did not respond to your message.", "thoughts": "Error during action."}
            party_raw_response = json.dumps(party_action_json)

        self.last_party_action_json = party_action_json
        
        self._record_turn(
            actor_role=next_acting_party_id,
            prompt_or_payload=party_prompt_messages,
            raw_response=party_raw_response,
            parsed_response_json=party_action_json,
            recipient_party_id=None,
            actor_identifier=next_acting_party_id
        )

        return next_acting_party_id, party_action_json, party_raw_response, False

    def _finalize_negotiation_results(self) -> None:
        if self.results is not None:
            return

        print("\n--- Finalizing Negotiation Results ---")
        accepted = False
        if self.current_offer is not None:
            if self.offer_proposed_by == "seller" and self.buyer_accepted:
                accepted = True
            elif self.offer_proposed_by == "buyer" and self.seller_accepted:
                accepted = True
        
        final_outcome_for_payoff = self.current_offer if accepted else None

        if accepted:
            print(f"Agreement reached! Offer: {final_outcome_for_payoff}")
        else:
            print("No agreement reached.")

        effective_turns_for_payoff = self.turn 

        payoff_A, payoff_B = self.scenario.get_payoffs(final_outcome_for_payoff, effective_turns_for_payoff)
        print(f"Final Payoffs: Party A ({self.party_A.role}) = {payoff_A}, Party B ({self.party_B.role}) = {payoff_B}")

        self.results = {
            "scenario": self.scenario.name,
            "party_A": self.party_A.name,
            "party_B": self.party_B.name,
            "mediator": "external_mediator_tool_user",
            "accepted": accepted,
            "final_offer": final_outcome_for_payoff,
            "payoff_A": payoff_A,
            "payoff_B": payoff_B,
            "turns_taken_for_payoff_calc": effective_turns_for_payoff,
            "total_transcript_entries": len(self.transcript),
            "transcript": self.transcript,
        }

    def score_by_payoff(self, party: PartyAgent) -> float:
        if not self.results:
            self._finalize_negotiation_results()
            if not self.results:
                 raise ValueError("Results not found and could not be finalized. Run the negotiation steps first.")

        payoff = self.results["payoff_A"] if party.role == "seller" else self.results["payoff_B"]
        sigma = self.scenario.outcome_sigma
        if sigma == 0: return 0.0
        return float(payoff / sigma) if payoff is not None else 0.0


    def score_by_agent_eval(self, party: PartyAgent) -> float:
        if not self.results:
            self._finalize_negotiation_results()
            if not self.results:
                raise ValueError("Results not found for agent evaluation. Run negotiation steps first.")

        party_id = "seller" if party == self.party_A else "buyer"
        private_context = self.scenario.get_private_context(party.role)
        private_context_str = self.scenario.format_private_context(private_context)
        
        final_offer_display = "Negotiation concluded with no agreement reached."
        if self.results["accepted"] and self.results["final_offer"]:
            final_offer_display = json.dumps(self.results["final_offer"])

        base_eval_context = {
            "role": party.role,
            "scenario_name": self.scenario.name,
            "private_context": private_context_str,
            "final_offer": final_offer_display,
            "time_taken": f"{self.results.get('turns_taken_for_payoff_calc', self.turn)} rounds/exchanges",
        }
        base_eval_prompt_str = load_template("evaluator_prompt.txt")
        system_prompt_content = render_template(base_eval_prompt_str, base_eval_context)

        messages_for_evaluator = [{"role": "system", "content": system_prompt_content}]
        
        chat_history_for_eval = []
        for entry in self.results.get("transcript", []):
            if entry['role'] == "mediator" and entry['recipient'] == party_id:
                 chat_history_for_eval.append({"role": "agent", "content": entry['response'].get('message', entry['raw_response'])})
            elif entry['actor_id'] == party_id and entry['recipient'] is None:
                eval_content_parts = []
                if 'thoughts' in entry['response'] and entry['response']['thoughts']:
                     eval_content_parts.append(f"My thoughts: {entry['response'].get('thoughts', '')}")
                eval_content_parts.append(f"My message to mediator: {entry['response'].get('message', '')}")
                eval_content_parts.append(f"My proposed action: {entry['response'].get('action', '')}")
                if entry['response'].get('price') is not None:
                    eval_content_parts.append(f"My proposed price: {entry['response'].get('price')}")
                chat_history_for_eval.append({"role": entry['role'], "content": "\n".join(eval_content_parts)})
        
        messages_for_evaluator.extend(chat_history_for_eval)

        prompt_for_evaluator = self.tokenizer.apply_chat_template(
            messages_for_evaluator,
            tokenize=False
        )

        evaluator_response_raw = party.act([{"role": "system", "content": prompt_for_evaluator}]).content
        evaluator_response_json = extract_json_block(evaluator_response_raw)
        if evaluator_response_json and isinstance(evaluator_response_json.get("rating"), (int, float)):
            return float(evaluator_response_json["rating"]) / 10.0
        else:
            print(f"Warning: Missing or invalid 'rating' key in evaluator JSON for {party_id}. Response: {evaluator_response_raw}")
            return 0.0
