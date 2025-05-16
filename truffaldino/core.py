import random
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Literal

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

    def __init__(self, scenario: BaseScenario, party_A: PartyAgent, party_B: PartyAgent, mediator: MediatorAgent, tokenizer_name_or_path: str = "gpt2"):
        self.scenario = scenario
        self.party_A = party_A
        self.party_B = party_B
        self.mediator = mediator

        self.transcript: List[Dict[str, Any]] = [] # Store structured turn info
        self.turn: int = 0
        self.current_offer: Optional[Dict[str, Any]] = None
        self.offer_proposed_by: Optional[Literal["seller", "buyer"]] = None
        self.seller_accepted: bool = False
        self.buyer_accepted: bool = False
        self.last_mediator_message_to: Dict[Literal["seller", "buyer"], str] = {"seller": "", "buyer": ""}

        self.results: List[Dict[str, Any]] = []
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # Common practice for models like GPT-2

        # Define the Jinja chat template
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
                "{% if message.role == 'system' %}"
                    "{{ message.content \n}}"
                "{% elif message.role == 'seller' %}"
                    "\n\n**Party A (to Agent):**\n{{ message.content }}"
                "{% elif message.role == 'buyer' %}"
                    "\n\n**Party B (to Agent):**\n{{ message.content }}"
                 "{% elif message.role == 'agent' %}"
                    "\n\n**Agent (to {{ message.recipient }}):**\n{{ message.content }}"
                "{% else %}"
                    "\n\n**{{ message.role }}:**\n{{ message.content }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                 "{% if messages[-1].role == 'buyer' or messages[-1].role == 'seller' %}"
                    "\n\n**Agent (to {{ messages[-1].next_recipient }}):**\n"                   
                 "{% endif %}"
            "{% endif %}"
        )

    def _get_party(self, party_id: Literal["seller", "buyer"]) -> PartyAgent:
        return self.party_A if party_id == "seller" else self.party_B

    def _prepare_chat_for_recipient(self, recipient: Literal["seller", "buyer", "agent"]) -> List[Dict[str, str]]:
        chat_messages = []
        for entry in self.transcript:
            if recipient == "agent":
                if entry['recipient'] is not None:
                    chat_messages.append({"role": "agent", "content": f"{entry['raw_response']}", "next_recipient": entry['next_recipient']})
                else:
                    content = []
                    content.append(f"  Turn: {entry['turn']}")
                    content.append(f"  Message: {entry['response'].get('message', '')}")
                    content.append(f"  Action: {entry['response'].get('action', '')}")
                    content.append(f"  Price: {entry['response'].get('price', '')}")
                    chat_messages.append({"role": f"{entry['role']}", "content": "\n".join(content)})
            elif recipient == entry['recipient']:
                content = []
                content.append(f"  Turn: {entry['turn']}")
                content.append(f"  Message: {entry['response'].get('message', '')}")
                content.append(f"  Action: {entry['response'].get('action', '')}")
                content.append(f"  Price: {entry['response'].get('price', '')}")
                chat_messages.append({"role": f"{entry['role']}", "content": "\n".join(content)})
            elif entry['actor_id'] == recipient and entry['recipient'] is None:
                content = []
                content.append(f"  Turn: {entry['turn']}")
                content.append(f"  Thoughts: {entry['response'].get('thoughts', '')}")
                content.append(f"  Message: {entry['response'].get('message', '')}")
                content.append(f"  Action: {entry['response'].get('action', '')}")
                content.append(f"  Price: {entry['response'].get('price', '')}")
                chat_messages.append({"role": f"{entry['role']}", "content": "\n".join(content)})
            else:
                pass
        
        return chat_messages if chat_messages else [{"role": "system", "content": "No messages yet."}]

    def _build_party_prompt(self, party_id: Literal["seller", "buyer"]) -> str:
        party = self._get_party(party_id)
        private_context_dict = self.scenario.get_private_context(party.role)
        private_context_str = self.scenario.format_private_context(private_context_dict)

        # Load the base prompt for the party
        base_prompt_template_str = load_template("party_prompt.txt")
        # Render any specific fields for the base prompt (excluding transcript)
        base_prompt_context = {
            "role": party.role,
            "scenario_name": self.scenario.name,
            "private_context": private_context_str,
            "current_offer_price": self.current_offer.get('price') if self.current_offer else None,
            "current_offer_by": self.offer_proposed_by,
        }
        system_prompt_content = render_template(base_prompt_template_str, base_prompt_context)

        # Prepare messages for chat template
        messages = [
            {"role": "system", "content": system_prompt_content}
        ]
        
        chat_history = self._prepare_chat_for_recipient(party_id)
        messages.extend(chat_history)

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _build_mediator_prompt(self, party_id: Literal["seller", "buyer"]) -> str:
        active_party = self._get_party(party_id)
        inactive_party = self._get_party("buyer" if party_id == "seller" else "seller")

        mediator_private_context_dict = self.scenario.get_private_context("agent")
        mediator_private_context_str = self.scenario.format_private_context(mediator_private_context_dict)

        base_prompt_template_str = load_template("mediator_prompt.txt")
        base_prompt_context = {
            "role": "mediator",
            "private_context": mediator_private_context_str,
            "scenario_name": self.scenario.name,
            "party_A_name": self.party_A.name,
            "party_A_role": self.party_A.role,
            "party_B_name": self.party_B.name,
            "party_B_role": self.party_B.role,
            "active_party_name": active_party.name,
            "active_party_role": active_party.role,
            "inactive_party_name": inactive_party.name,
            "inactive_party_role": inactive_party.role,
        }
        system_prompt_content = render_template(base_prompt_template_str, base_prompt_context)

        # Prepare messages for chat template
        messages = [
            {"role": "system", "content": system_prompt_content}
        ]

        chat_history = self._prepare_chat_for_recipient("agent")
        messages.extend(chat_history)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _record_turn(
            self,
            actor: BaseAgent, 
            role: str, 
            prompt: str, 
            message: str, 
            action_json: Optional[Dict[str, Any]], 
            recipient: Optional[Literal["seller", "buyer", "agent"]], 
            actor_id: Optional[Literal["seller", "buyer", "agent"]]):
        response_dict = extract_json_block(message)
        self.transcript.append({
            "turn": self.turn,
            "actor": actor.name,
            "actor_id": actor_id,
            "role": role,
            "prompt": prompt,
            "response": response_dict,
            "raw_response": message,
            "action_json": action_json,
            "recipient": recipient, # Who the mediator sent the message to
            "next_recipient": "buyer" if actor.role == "seller" else "agent" if actor.role == "agent" else "seller", # Who the mediator will send the message to next
        })

    def _update_state_from_action(self, action: Dict[str, Any], party_id: Literal["seller", "buyer"]):
        action_type = action.get("action")

        if action_type == "accept":
            if party_id == "seller":
                self.seller_accepted = True
            else:
                self.buyer_accepted = True
            # Acceptance requires a standing offer from the *other* party
            if not self.current_offer or self.offer_proposed_by == party_id:
                 print(f"Warning: Party {party_id} accepted but no valid offer from other party exists. Treating as reject.")
                 self.seller_accepted = False
                 self.buyer_accepted = False

        elif action_type == "offer":
            if self.scenario.validate_offer(action):
                self.current_offer = action # Store the whole offer dict
                self.offer_proposed_by = party_id
                # Reset acceptance flags on new offer
                self.seller_accepted = False
                self.buyer_accepted = False
            else:
                print(f"Warning: Invalid offer structure from {party_id}: {action}")
                # Treat invalid offer effectively as a reject/no-op
                pass

        elif action_type == "counteroffer":
            # Reset acceptance flags if anyone rejects
            self.seller_accepted = False
            self.buyer_accepted = False
            # Handle counter-offer if present
            counter_offer = action.get("price")
            if counter_offer and isinstance(counter_offer, dict):
                 if self.scenario.validate_offer(counter_offer):
                     self.current_offer = counter_offer
                     self.offer_proposed_by = party_id
                 else:
                     print(f"Warning: Invalid counter-offer structure from {party_id}: {counter_offer}")
            # If simple reject, the current offer remains (if any) but is no longer accepted

        elif action_type == 'reject':
            # Reset acceptance flags if anyone rejects
            self.seller_accepted = False
            self.buyer_accepted = False
        
        elif action_type == 'inquiry':
            # Do nothing
            pass

        else:
            print(f"Warning: Unknown action type '{action_type}' from {party_id}")
            # Treat unknown action as reject/no-op
            pass

    def _is_finished(self) -> bool:
        # Check for agreement first
        agreement_reached = False
        if self.current_offer is not None:
            if self.offer_proposed_by == "seller" and self.buyer_accepted:
                agreement_reached = True
            elif self.offer_proposed_by == "buyer" and self.seller_accepted:
                agreement_reached = True

        if agreement_reached:
            return True

        # Check if max turns reached
        # Turn count increments *after* mediator speaks, so check >= max_turns * 2
        return self.turn >= self.scenario.n_turns * 2

    def run(self, n_runs: int = 1, seed: Optional[int] = None, md_log_path: Optional[str] = None, json_log_path: Optional[str] = None, debug: bool = False) -> Dict[str, Any]:
        """Runs the negotiation process until completion, optionally logging to files."""

        self.results = []
        for run_idx in range(n_runs):
            if seed is not None:
                random.seed(seed)
            seed = random.randint(0, 2**32 - 1)
            self.results.append(self.run_single(seed, f"{md_log_path}_{run_idx}.md", f"{json_log_path}_{run_idx}.json", debug))
        return self.results

    def run_single(self, seed: int, md_log_path: Optional[str] = None, json_log_path: Optional[str] = None, debug: bool = False) -> Dict[str, Any]:
        # Reset state for potentially multiple runs
        self.transcript = []
        self.turn = 0
        self.current_offer = None
        self.offer_proposed_by = None
        self.seller_accepted = False
        self.buyer_accepted = False
        self.last_mediator_message_to = {"seller": "", "buyer": ""}

        # Determine starting party
        active_party_id: Literal["seller", "buyer"] = random.choice(["seller", "buyer"])

        md_log_file = None
        if md_log_path:
            try:
                md_log_file = open(md_log_path, 'w', encoding='utf-8')
                md_log_file.write(f"# Negotiation Log: {self.scenario.name}\n\n")
                md_log_file.write(f"- Party A ({self.party_A.role}): {self.party_A.name}\n")
                md_log_file.write(f"- Party B ({self.party_B.role}): {self.party_B.name}\n")
                md_log_file.write(f"- Mediator: {self.mediator.name}\n")
                md_log_file.write(f"- Seed: {seed}\n")
                md_log_file.write(f"- Starting Party: {active_party_id}\n")
                md_log_file.write(f"- Max turns per party: {self.scenario.n_turns}\n\n")
                md_log_file.write(f"- Negotiation State:\n")
                md_log_file.write(f"  {self.scenario.negotiation_state}\n")
                md_log_file.write("---\n\n")
            except IOError as e:
                print(f"Warning: Could not open markdown log file {md_log_path}: {e}")
                md_log_file = None # Ensure it's None if opening failed

        print(f"--- Starting Negotiation: {self.scenario.name} ---")
        print(f"Party A ({self.party_A.role}): {self.party_A.name}")
        print(f"Party B ({self.party_B.role}): {self.party_B.name}")
        print(f"Mediator: {self.mediator.name}")
        print(f"Starting Party: {active_party_id}")
        print(f"Max turns per party: {self.scenario.n_turns}")
        print("---")

        while not self._is_finished():
            current_party = self._get_party(active_party_id)
            inactive_party_id = "buyer" if active_party_id == "seller" else "seller"
            turn_num_display = self.turn // 2 + 1
            party_turn_marker = 1 if active_party_id == 'seller' else 2
            if debug:
                print(f"\n--- Turn {turn_num_display}.{party_turn_marker}: {current_party.name}'s move (to Mediator) ---")
            if md_log_file:
                 md_log_file.write(f"## Turn {turn_num_display}.{party_turn_marker}: {current_party.name} -> Mediator\n\n")

            # 1. Get Party Action
            party_prompt = self._build_party_prompt(active_party_id)
            try:
                party_raw_response = current_party.act(party_prompt)
                party_action_json = extract_json_block(party_raw_response)
                if not party_action_json or not isinstance(party_action_json.get("action"), str):
                    raise ParseError("Missing or invalid 'action' key in party JSON.")
            except (ParseError, TypeError, Exception) as e:
                print(f"Error getting/parsing action from {current_party.name}: {e}")
                print("Treating as REJECT.")
                party_action_json = {"action": "reject"} # Default to reject on error
                party_raw_response = f"Error: {e}\nDefaulting to: {json.dumps(party_action_json)}"

            # Log party message before recording internally
            if md_log_file:
                md_log_file.write(f"**{current_party.name} ({current_party.role}) says:**\n")
                md_log_file.write(f"{party_prompt}")
                md_log_file.write(f"{party_raw_response}")

            self._record_turn(current_party, current_party.role, party_prompt, party_raw_response, party_action_json, recipient=None, actor_id=active_party_id) # Party -> Mediator
            if debug:
                print(f"{current_party.name} proposed action: {party_action_json}")

            # 2. Mediator Relays Action
            if debug:
                print(f"--- Turn {turn_num_display}.{party_turn_marker}: Mediator relays to {inactive_party_id} ---")
            if md_log_file:
                md_log_file.write(f"## Turn {turn_num_display}.{party_turn_marker}: Mediator -> {self._get_party(inactive_party_id).name}\n\n")

            mediator_prompt = self._build_mediator_prompt(active_party_id)
            try:
                mediator_raw_response = self.mediator.act(mediator_prompt)
                mediator_action_json = extract_json_block(mediator_raw_response)
                if not mediator_action_json or not isinstance(mediator_action_json.get("action"), str):
                     raise ParseError("Missing or invalid 'action' key in mediator JSON.")
                # Crucially, the mediator *should* be relaying the *party's* action
                # Here we parse the *mediator's* message to log what *it* sent
            except (ParseError, TypeError, Exception) as e:
                print(f"Error getting/parsing action from {self.mediator.name}: {e}")
                print(f"Attempting to relay original party action {party_action_json} directly.")
                mediator_action_json = party_action_json # Fallback to party's action
                mediator_raw_response = f"Error processing mediator response: {e}\nRelaying party action: {json.dumps(mediator_action_json)}"

            # Log mediator message before recording internally
            if md_log_file:
                md_log_file.write(f"**{self.mediator.name} (Mediator) says to {self._get_party(inactive_party_id).name}:**\n")
                md_log_file.write(f"```text\n{mediator_prompt}\n```\n")
                md_log_file.write(f"```text\n{mediator_raw_response}\n```\n")
                md_log_file.write(f"**Action Parsed/Relayed:** `{json.dumps(mediator_action_json)}`\n\n")
                md_log_file.write("---\n\n") # Separator between full turns

            # 3. Update State based on the *original party's intended action*
            self._update_state_from_action(party_action_json, active_party_id)

            # 4. Record Mediator's turn and store message for next party prompt
            self._record_turn(self.mediator, "mediator", mediator_prompt, mediator_raw_response, mediator_action_json, recipient=inactive_party_id, actor_id=active_party_id)
            self.last_mediator_message_to[inactive_party_id] = mediator_raw_response
            if debug:
                print(f"Mediator message to {inactive_party_id}: {mediator_action_json}")

            # 5. Advance turn and switch party
            self.turn += 1
            active_party_id = inactive_party_id

            # Check for early finish after state update
            if self._is_finished():
                break

        # 6. Determine Final Outcome
        print("\n--- Negotiation Finished ---")
        final_outcome: Optional[Dict[str, Any]] = None
        # Determine acceptance based on the logic in _is_finished
        accepted = False
        if self.current_offer is not None:
            if self.offer_proposed_by == "seller" and self.buyer_accepted:
                accepted = True
            elif self.offer_proposed_by == "buyer" and self.seller_accepted:
                accepted = True

        if md_log_file:
            md_log_file.write("## Final Outcome\n\n")

        if accepted:
            final_outcome = self.current_offer
            print(f"Agreement reached! Offer: {final_outcome}")
            if md_log_file:
                md_log_file.write(f"**Agreement reached!**\n")
                md_log_file.write(f"- Final Offer: `{json.dumps(final_outcome)}`\n")
        else:
            print("No agreement reached.")
            if md_log_file:
                md_log_file.write("**No agreement reached.**\n")

        payoff_A, payoff_B = self.scenario.get_payoffs(final_outcome, self.turn // 2)
        print(f"Final Payoffs: Party A ({self.party_A.role}) = {payoff_A}, Party B ({self.party_B.role}) = {payoff_B}")
        if md_log_file:
            md_log_file.write(f"- Final Payoff A ({self.party_A.role}): {payoff_A}\n")
            md_log_file.write(f"- Final Payoff B ({self.party_B.role}): {payoff_B}\n")
            md_log_file.write(f"- Turns taken (total agent messages): {self.turn}\n")


        self.results = {
            "seed": seed,
            "scenario": self.scenario.name,
            "party_A": self.party_A.name,
            "party_B": self.party_B.name,
            "mediator": self.mediator.name,
            "accepted": accepted,
            "final_offer": final_outcome,
            "payoff_A": payoff_A,
            "payoff_B": payoff_B,
            "turns_taken": self.turn,
            "transcript": self.transcript,
        }


        eval_party_A = self.score_by_agent_eval(self.party_A)
        eval_party_B = self.score_by_agent_eval(self.party_B)

        self.results["eval_party_A"] = eval_party_A
        self.results["eval_party_B"] = eval_party_B
        self.results['normalized_payoff_A'] = self.score_by_payoff(self.party_A)
        self.results['normalized_payoff_B'] = self.score_by_payoff(self.party_B)

        # Close markdown log file if open
        if md_log_file:
            md_log_file.write(f"## Evaluation\n\n")
            md_log_file.write(f"**Party A ({self.party_A.role}) evaluation:** {eval_party_A}\n")
            md_log_file.write(f"**Party B ({self.party_B.role}) evaluation:** {eval_party_B}\n")

            try:
                md_log_file.close()
            except IOError as e:
                print(f"Warning: Could not close markdown log file {md_log_path}: {e}")


        # Save results to JSON log file if path provided
        if json_log_path:
            try:
                with open(json_log_path, 'w', encoding='utf-8') as f_json:
                    json.dump(self.results, f_json, indent=2)
            except IOError as e:
                print(f"Warning: Could not write JSON log file {json_log_path}: {e}")


        return self.results 

    
    def score_by_payoff(self, party: PartyAgent) -> float:
        """
        Scores the payoff for a given role.
        """
        if not self.results:
            raise ValueError("Results not found. Run the negotiation first.")
        outcome = self.results["final_offer"]
        seller_payoff, buyer_payoff = self.scenario.get_payoffs(outcome, self.turn // 2)

        sigma = self.scenario.outcome_sigma

        if party.role == "seller":
            return float(seller_payoff/sigma)
        elif party.role == "buyer":
            return float(buyer_payoff/sigma)
        else:
            raise ValueError(f"Unknown role for scoring: {party.role}")


    def score_by_agent_eval(self, party: PartyAgent) -> float:
        """
        Scores the payoff for a given role.
        """
        if not self.results:
            raise ValueError("Results not found. Run the negotiation first.")
        party_id = "seller" if party == self.party_A else "buyer"
        private_context = self.scenario.get_private_context(party.role)
        private_context_str = self.scenario.format_private_context(private_context)
        chat_history = self._prepare_chat_for_recipient(party_id)

        final_offer = "Negotiation concluded with no agreement reached" if not self.results["accepted"] else self.results["final_offer"]

        base_eval_context = {
            "role": party.role,
            "scenario_name": self.scenario.name,
            "private_context": private_context_str,
            "final_offer": final_offer,
            "time_taken": f"{self.turn // 2} days",
        }
        base_eval_prompt_str = load_template("evaluator_prompt.txt")
        system_prompt_content = render_template(base_eval_prompt_str, base_eval_context)

        # Prepare messages for chat template
        messages = [
            {"role": "system", "content": system_prompt_content}
        ]
        
        # Get the conversation history formatted for this party for evaluation context
        # This should use the party-specific view (roles 'user' and 'agent')
        chat_history = self._prepare_chat_for_recipient(party_id) # party_id is 'seller' or 'buyer'
        messages.extend(chat_history)

        evaluator_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) # No generation prompt for eval

        evaluator_response = party.act(evaluator_prompt)
        evaluator_response_json = extract_json_block(evaluator_response)
        if not evaluator_response_json or not isinstance(evaluator_response_json.get("rating"), (int, float)):
            raise ParseError("Missing or invalid 'rating' key in evaluator JSON.")
        return evaluator_response_json["rating"]/10.0
        