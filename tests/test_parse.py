import pytest
from truffaldino.parse import extract_json_block, ParseError

# Test cases for extract_json_block
VALID_CASES = [
    ("Some text before\n```json\n{\"key\": \"value\", \"number\": 123}\n```\nSome text after", {"key": "value", "number": 123}),
    ("```json\n{\"action\": \"offer\", \"price\": 500}\n```", {"action": "offer", "price": 500}),
    ("No JSON block here", None),
    ("```json\n\n```", None), # Empty block
    ("   ```json\n   {\"nested\": {\"a\": 1}}   \n```   ", {"nested": {"a": 1}}), # Whitespace variation
    ("```JSON\n{\"case\": \"insensitive\"}\n```", {"case": "insensitive"}), # Case insensitive marker
    # Fallback case: raw JSON string
    ('{"fallback": true, "value": 42}', {"fallback": True, "value": 42}),
    ('[1, 2, 3]', None), # Fallback should reject non-dict JSON
    ('"just a string"', None), # Fallback should reject non-dict JSON
]

INVALID_CASES = [
    ("```json\n{\"key\": \"value\", \"number\": 123\n```", "Invalid JSON detected: Unterminated string starting at"), # Missing closing brace
    ("```json\n{\"key\": \"value\" \"number\": 123}\n```", "Invalid JSON detected: Expecting ',' delimiter:"), # Missing comma
    ("```json\n{\"key\": value}\n```", "Invalid JSON detected: Expecting value:"), # Unquoted string value
    ("```json\n{'key': 'value'}\n```", "Invalid JSON detected: Expecting property name enclosed in double quotes:"), # Single quotes
    ("```json\n{\"key\": [1, 2, ]}\n```", "Invalid JSON detected: Trailing comma found:"), # Trailing comma in list
    # Non-dict JSON is now a ParseError if inside markers
    ("```json\n[1, 2, 3]\n```", "Parsed JSON is not a dictionary: <class 'list'>"),
]

@pytest.mark.parametrize("text, expected_output", VALID_CASES)
def test_extract_json_block_valid(text, expected_output):
    assert extract_json_block(text) == expected_output

@pytest.mark.parametrize("text, error_message_start", INVALID_CASES)
def test_extract_json_block_invalid(text, error_message_start):
    with pytest.raises(ParseError) as excinfo:
        extract_json_block(text)
    assert str(excinfo.value).startswith(error_message_start)


# Edge case: Multiple JSON blocks (should only extract the first)
def test_extract_json_block_multiple():
    text = ("First block:\n```json\n{\"first\": 1}\n```\n" +
            "Second block:\n```json\n{\"second\": 2}\n```")
    assert extract_json_block(text) == {"first": 1}

# Edge case: Malformed markers
def test_extract_json_block_malformed_markers():
    text = "```json\n{\"key\": \"value\"}"
    # No closing marker, should ideally not match or handle gracefully
    # Current regex requires closing marker, so this should be None
    assert extract_json_block(text) is None

    text = "```\n{\"key\": \"value\"}\n```"
    # Missing language identifier
    assert extract_json_block(text) is None 