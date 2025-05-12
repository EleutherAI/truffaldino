import json
import re
from typing import Dict, Any, Optional

class ParseError(Exception):
    """Custom exception for parsing errors."""
    pass

def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extracts the first JSON block enclosed in ```json ... ```.

    Args:
        text: The string potentially containing the JSON block.

    Returns:
        A dictionary parsed from the JSON block, or None if no block is found.

    Raises:
        ParseError: If a JSON block is found but is invalid.
    """
    # Regex to find the first ```json ... ``` block
    # It handles potential leading/trailing whitespace and newlines around the block
    match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL | re.IGNORECASE)

    if not match:
        # Fallback: maybe the LLM just returned JSON directly?
        try:
            # Attempt to parse the whole string as JSON
            # Be cautious with this fallback, might misinterpret plain text
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            else:
                # If it parses but isn't a dict, it's not what we want
                return None
        except json.JSONDecodeError:
            # If parsing the whole string fails, definitely no valid JSON block
            return None # No JSON block found

    json_str = match.group(1).strip()

    if not json_str:
        # Found the markers but the block is empty
        return None

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
             raise ParseError(f"Parsed JSON is not a dictionary: {type(data)}")
        return data
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON detected: {e}\nContent:\n{json_str}") from e 