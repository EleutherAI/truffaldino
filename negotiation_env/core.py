import random
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Literal

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

    def __init__(self, scenario: BaseScenario, party_A: PartyAgent, party_B: PartyAgent, mediator: MediatorAgent):
        self.scenario = scenario
        self.party_A = party_A
        self.party_B = party_B
        self.mediator = mediator

        self.transcript: List[Dict[str, Any]] = [] # Store structured turn info
        self.turn: int = 0
        self.current_offer: Optional[Dict[str, Any]] = None
        self.offer_proposed_by: Optional[Literal["A", "B"]] = None
        self.party_A_accepted: bool = False
        self.party_B_accepted: bool = False
        self.last_mediator_message_to: Dict[Literal["A", "B"], str] = {"A": "", "B": ""}

    def _get_party(self, party_id: Literal["A", "B"]) -> PartyAgent:
        return self.party_A if party_id == "A" else self.party_B

    def _format_transcript_for_prompt(self, recipient: Literal["A", "B", "Mediator"]) -> str:
        lines = []
        for entry in self.transcript:
            if recipient == "Mediator":
                lines.append(f"Turn {entry['turn']} - {entry['actor']} ({entry['role']}):")
                lines.append(f"  Message: {entry['message']}")
                if entry['action_json']:
                    lines.append(f"  Action: {json.dumps(entry['action_json'])}")
            elif recipient == entry['recipient']:
                 lines.append(f"Turn {entry['turn']} - From Mediator:")
                 lines.append(f"  {entry['message']}")
                 if entry['action_json']:
                    lines.append(f"  Action Conveyed: {json.dumps(entry['action_json'])}")

        return "\n".join(lines) if lines else "No messages yet."

    def _build_party_prompt(self, party_id: Literal["A", "B"]) -> str:
        party = self._get_party(party_id)
        private_context_dict = self.scenario.get_private_context(party.role)
        private_context_str = self.scenario.format_private_context(private_context_dict)

        context = {
            "role": party.role,
            "scenario_name": self.scenario.name,
            "private_context": private_context_str,
            "current_offer_price": self.current_offer.get('price') if self.current_offer else None,
            "current_offer_by": self.offer_proposed_by,
            "transcript": self._format_transcript_for_prompt(party_id),
        }
        template_str = load_template("party_prompt.txt")
        return render_template(template_str, context)

    def _build_mediator_prompt(self, party_id: Literal["A", "B"], party_message_raw: str, party_message_json: Dict[str, Any]) -> str:
        active_party = self._get_party(party_id)
        inactive_party = self._get_party("B" if party_id == "A" else "A")

        context = {
            "scenario_name": self.scenario.name,
            "party_A_name": self.party_A.name,
            "party_A_role": self.party_A.role,
            "party_B_name": self.party_B.name,
            "party_B_role": self.party_B.role,
            "transcript": self._format_transcript_for_prompt("Mediator"),
            "active_party_name": active_party.name,
            "active_party_role": active_party.role,
            "inactive_party_name": inactive_party.name,
            "inactive_party_role": inactive_party.role,
            "party_message_raw": party_message_raw,
            "party_message_json": json.dumps(party_message_json, indent=2),
        }
        template_str = load_template("mediator_prompt.txt")
        return render_template(template_str, context)

    def _record_turn(self, actor: BaseAgent, role: str, message: str, action_json: Optional[Dict[str, Any]], recipient: Optional[Literal["A", "B"]]):
        self.transcript.append({
            "turn": self.turn,
            "actor": actor.name,
            "role": role,
            "message": message,
            "action_json": action_json,
            "recipient": recipient, # Who the mediator sent the message to
        })

    def _update_state_from_action(self, action: Dict[str, Any], party_id: Literal["A", "B"]):
        action_type = action.get("action")

        if action_type == "accept":
            if party_id == "A":
                self.party_A_accepted = True
            else:
                self.party_B_accepted = True
            # Acceptance requires a standing offer from the *other* party
            if not self.current_offer or self.offer_proposed_by == party_id:
                 print(f"Warning: Party {party_id} accepted but no valid offer from other party exists. Treating as reject.")
                 self.party_A_accepted = False
                 self.party_B_accepted = False

        elif action_type == "offer":
            if self.scenario.validate_offer(action):
                self.current_offer = action # Store the whole offer dict
                self.offer_proposed_by = party_id
                # Reset acceptance flags on new offer
                self.party_A_accepted = False
                self.party_B_accepted = False
            else:
                print(f"Warning: Invalid offer structure from {party_id}: {action}")
                # Treat invalid offer effectively as a reject/no-op
                pass

        elif action_type == "reject":
            # Reset acceptance flags if anyone rejects
            self.party_A_accepted = False
            self.party_B_accepted = False
            # Handle counter-offer if present
            counter_offer = action.get("counter_offer")
            if counter_offer and isinstance(counter_offer, dict):
                 if self.scenario.validate_offer(counter_offer):
                     self.current_offer = counter_offer
                     self.offer_proposed_by = party_id
                 else:
                     print(f"Warning: Invalid counter-offer structure from {party_id}: {counter_offer}")
            # If simple reject, the current offer remains (if any) but is no longer accepted

        else:
            print(f"Warning: Unknown action type '{action_type}' from {party_id}")
            # Treat unknown action as reject/no-op
            pass

    def _is_finished(self) -> bool:
        # Check for agreement first
        agreement_reached = False
        if self.current_offer is not None:
            if self.offer_proposed_by == "A" and self.party_B_accepted:
                agreement_reached = True
            elif self.offer_proposed_by == "B" and self.party_A_accepted:
                agreement_reached = True

        if agreement_reached:
            return True

        # Check if max turns reached
        # Turn count increments *after* mediator speaks, so check >= max_turns * 2
        return self.turn >= self.scenario.n_turns * 2

    def run(self, seed: Optional[int] = None, md_log_path: Optional[str] = None, json_log_path: Optional[str] = None) -> Dict[str, Any]:
        """Runs the negotiation process until completion, optionally logging to files."""
        if seed is not None:
            random.seed(seed)

        # Reset state for potentially multiple runs
        self.transcript = []
        self.turn = 0
        self.current_offer = None
        self.offer_proposed_by = None
        self.party_A_accepted = False
        self.party_B_accepted = False
        self.last_mediator_message_to = {"A": "", "B": ""}

        # Determine starting party
        active_party_id: Literal["A", "B"] = random.choice(["A", "B"])

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
            inactive_party_id = "B" if active_party_id == "A" else "A"
            turn_num_display = self.turn // 2 + 1
            party_turn_marker = 1 if active_party_id == 'A' else 2
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
                md_log_file.write(f"```text\n{party_raw_response}\n```\n")
                md_log_file.write(f"**Action Parsed:** `{json.dumps(party_action_json)}`\n\n")

            self._record_turn(current_party, current_party.role, party_raw_response, party_action_json, recipient=None) # Party -> Mediator
            print(f"{current_party.name} proposed action: {party_action_json}")

            # 2. Mediator Relays Action
            print(f"--- Turn {turn_num_display}.{party_turn_marker}: Mediator relays to {inactive_party_id} ---")
            if md_log_file:
                md_log_file.write(f"## Turn {turn_num_display}.{party_turn_marker}: Mediator -> {self._get_party(inactive_party_id).name}\n\n")

            mediator_prompt = self._build_mediator_prompt(active_party_id, party_raw_response, party_action_json)
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
                md_log_file.write(f"```text\n{mediator_raw_response}\n```\n")
                md_log_file.write(f"**Action Parsed/Relayed:** `{json.dumps(mediator_action_json)}`\n\n")
                md_log_file.write("---\n\n") # Separator between full turns

            # 3. Update State based on the *original party's intended action*
            self._update_state_from_action(party_action_json, active_party_id)

            # 4. Record Mediator's turn and store message for next party prompt
            self._record_turn(self.mediator, "mediator", mediator_raw_response, mediator_action_json, recipient=inactive_party_id)
            self.last_mediator_message_to[inactive_party_id] = mediator_raw_response
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
            if self.offer_proposed_by == "A" and self.party_B_accepted:
                accepted = True
            elif self.offer_proposed_by == "B" and self.party_A_accepted:
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

        payoff_A, payoff_B = self.scenario.get_payoffs(final_outcome)
        print(f"Final Payoffs: Party A ({self.party_A.role}) = {payoff_A}, Party B ({self.party_B.role}) = {payoff_B}")
        if md_log_file:
            md_log_file.write(f"- Final Payoff A ({self.party_A.role}): {payoff_A}\n")
            md_log_file.write(f"- Final Payoff B ({self.party_B.role}): {payoff_B}\n")
            md_log_file.write(f"- Turns taken (total agent messages): {self.turn}\n")

        # Close markdown log file if open
        if md_log_file:
            try:
                md_log_file.close()
            except IOError as e:
                print(f"Warning: Could not close markdown log file {md_log_path}: {e}")

        results = {
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

        # Save results to JSON log file if path provided
        if json_log_path:
            try:
                with open(json_log_path, 'w', encoding='utf-8') as f_json:
                    json.dump(results, f_json, indent=2)
            except IOError as e:
                print(f"Warning: Could not write JSON log file {json_log_path}: {e}")

        return results 