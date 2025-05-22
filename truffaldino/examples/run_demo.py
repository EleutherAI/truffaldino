import sys
import os
import argparse
import pprint
import json
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, List, Dict, Any
import asyncio

# Ensure the truffaldino package is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from truffaldino.agents import MediatorAgent
from truffaldino.tools.negotiation_tool import NegotiationTool
from truffaldino.llm_call import openrouter_llm_call

# --- Load API Key ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env file.")
    sys.exit(1)

# --- Constants ---
MAX_MEDIATOR_INTERACTIONS = 15

async def run_negotiation_loop(
    tool: NegotiationTool,
    instance_id: str,
    mediator_agent: MediatorAgent,
    initial_tool_params: Dict[str, Any],
    max_interactions: int,
    scenario_name: str
) -> List[Dict[str, str]]:
    mediator_chat_history: List[Dict[str, str]] = []

    tool_schema_dict = tool.get_openai_tool_schema().model_dump(exclude_none=True)
    
    system_prompt = f"""
Your role is to act as a neutral mediator in a negotiation.
At each step, you will be provided with the history of the conversation, including messages from parties (as 'user' role) and results from tool calls (as 'tool' role).
You MUST decide on the next action by calling the 'truffaldino_mediator_step' tool.

Tool Schema for 'truffaldino_mediator_step':
{json.dumps(tool_schema_dict, indent=2)}

Your response should be a JSON object containing a 'tool_calls' array. Each tool call should have this structure:
{{
  "tool_calls": [
    {{
      "id": "call_123",  // A unique identifier for this tool call
      "function": {{
        "name": "truffaldino_mediator_step",
        "arguments": "{{  // This is a JSON string containing the actual parameters
          \\"operation\\": \\"send_message_to_party\\",
          \\"mediator_json_payload_str\\": \\"{{\\\"action\\\": \\\"offer\\\", \\\"message\\\": \\\"Seller proposes $120.\\\", \\\"recipient\\\": \\\"buyer\\\", \\\"price\\\": 120, \\\"thinking\\\": \\\"Seller is trying to make a good offer.\\\"}}\\",
          \\"instance_id\\": \\"{instance_id}\\"
        }}"
      }}
    }}
  ]
}}

If the negotiation is just starting, you must call 'initialize_session' first. 
The initial setup for scenario '{scenario_name}' with seed {initial_tool_params.get('seed')} has been requested. You should call tools using the instance_id {instance_id}.
Once the initial session content has been populated, based on the initial setup result (which will appear as a 'tool' message), formulate your first message to one of the parties using the 'send_message_to_party' operation.
"""
    mediator_chat_history.append({"role": "system", "content": system_prompt})

    for i in range(max_interactions):
        print(f"\n--- LOOP: Mediator Interaction #{i + 1} ---")
        
        print(f"Calling mediator LLM ({mediator_agent.name})...")
        mediator_llm_response_message = mediator_agent.act(mediator_chat_history, temperature=0.5, tools=[tool_schema_dict])
        
        mediator_chat_history.append(mediator_llm_response_message.model_dump())
        print(f"Mediator LLM raw response: {mediator_llm_response_message.content}")

        tool_calls = mediator_llm_response_message.tool_calls
        if tool_calls is not None:
            for tool_call in tool_calls:
                tool_call_params = json.loads(tool_call.function.arguments)
                tool_call_response_str, _, tool_metrics = await tool.execute(instance_id, tool_call_params)
                mediator_chat_history.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id,
                    "name": "truffaldino_mediator_step", 
                    "content": tool_call_response_str
                })
                print(f"Tool output: {tool_call_response_str}")

                if tool_metrics.get("error"):
                    print(f"Tool execution error: {tool_metrics.get('error')}. Halting loop.")
                    mediator_chat_history.append({"role": "user", "content": f"System Note: Tool execution failed: {tool_metrics.get('error')}"})
                    break

                if tool_metrics.get("status") == "finished":
                    print("Negotiation finished according to tool.")
                    final_results = tool_metrics.get("results", {})
                    print("Final Results:")
                    pprint.pprint(final_results)
                    break 
                elif tool_metrics.get("status") == "continue" or tool_metrics.get("status") == "initialized":
                    next_party_id = tool_metrics.get("party_id_to_speak_to_mediator")
                    next_party_raw_response = tool_metrics.get("party_raw_response")
                    if next_party_id and next_party_raw_response is not None:
                        mediator_chat_history.append({"role": "user", "content": f"Message from {next_party_id} (to you, the mediator): {next_party_raw_response}"})
                    else:
                        print("Tool indicated continue but did not provide next party message. Halting.")
                        break
                else:
                    print(f"Tool returned unexpected status: {tool_metrics.get('status')}. Halting loop.")
                    break
        
        if i == max_interactions -1 : print ("Max interactions reached.")

    return mediator_chat_history

async def main_async(args):
    log_dir_base = os.path.join(project_root, "logs", "run_demo_tool_logs")
    json_log_dir = os.path.join(log_dir_base, "json")
    os.makedirs(json_log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_suffix = f"_seed{args.seed}"
    log_filename_base = f"{args.scenario}_{timestamp}{seed_suffix}"
    json_mediator_chat_log_path = os.path.join(json_log_dir, f"{log_filename_base}_mediator_chat.json")
    json_tool_results_log_path = os.path.join(json_log_dir, f"{log_filename_base}_tool_final_results.json")

    print(f"Running tool-based demo for scenario: {args.scenario} with seed: {args.seed}")
    print(f"JSON log for mediator chat: {json_mediator_chat_log_path}")
    print(f"JSON log for final tool results: {json_tool_results_log_path}")

    negotiation_tool = NegotiationTool()
    instance_id = await negotiation_tool.create()

    mediator_llm_params = {
        "model": args.mediator_model,
        "temperature": 0.6, 
        "max_tokens": 1500
    }
    mediator_agent = MediatorAgent(
        call_fn=lambda messages, **kwargs: openrouter_llm_call(messages, **{**mediator_llm_params, **kwargs}),
        name=f"MediatorBot ({args.mediator_model})"
    )

    initial_tool_params = {
        "operation": "initialize_session",
        "scenario_name": args.scenario,
        "seed": args.seed,
        "seller_llm_config": {"model": args.party_model},
        "buyer_llm_config": {"model": args.party_model}
    }

    final_mediator_chat_history = await run_negotiation_loop(
        tool=negotiation_tool,
        instance_id=instance_id,
        mediator_agent=mediator_agent,
        initial_tool_params=initial_tool_params,
        max_interactions=args.max_turns,
        scenario_name=args.scenario
    )

    try:
        with open(json_mediator_chat_log_path, 'w', encoding='utf-8') as f_json:
            json.dump(final_mediator_chat_history, f_json, indent=2)
        print(f"Mediator chat history saved to {json_mediator_chat_log_path}")
    except IOError as e:
        print(f"Warning: Could not write mediator chat log file {json_mediator_chat_log_path}: {e}")

    try:
        print("\n--- Retrieving Final Payoffs from Tool (Post-Loop) ---")
        payoff_params = {"operation": "get_payoffs"}
        tool_response_str, _, tool_metrics = await negotiation_tool.execute(instance_id, payoff_params)
        print(f"Final Payoff Tool Response: {tool_response_str}")
        if tool_metrics and not tool_metrics.get("error"):
            with open(json_tool_results_log_path, 'w', encoding='utf-8') as f_json:
                json.dump(tool_metrics.get("full_results", tool_metrics), f_json, indent=2)
            print(f"Final tool results saved to {json_tool_results_log_path}")
        else:
            print(f"Could not retrieve final results or tool error: {tool_metrics.get('error')}")
    except Exception as e:
         print(f"Error getting/saving final tool results: {e}")

    await negotiation_tool.release(instance_id)
    print(f"Negotiation tool instance {instance_id} released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a tool-based negotiation session loop.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="house_price",
        choices=["house_price"],
        help="Name of the negotiation scenario to run."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If None, will be random.")
    parser.add_argument("--max_turns", type=int, default=MAX_MEDIATOR_INTERACTIONS, help="Max number of mediator LLM interactions.")
    parser.add_argument("--mediator_model", type=str, default="google/gemini-2.5-flash-preview-05-20", help="Model name for the mediator agent on OpenRouter.")
    parser.add_argument("--party_model", type=str, default="qwen/qwen3-14b:free", help="Model name for party agents on OpenRouter.")
    args = parser.parse_args()

    if args.seed is not None and args.seed < 0:
        args.seed = None 
        print("Seed was negative, setting to None (random).")
    
    asyncio.run(main_async(args)) 