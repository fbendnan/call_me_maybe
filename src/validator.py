def json_state(text: str):
    """Return (is_valid, depth, in_string, escape, is_complete)."""
    depth = 0
    in_string = False
    escape = False
    last_non_whitespace = None

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
        if not ch.isspace():
            last_non_whitespace = ch

    if depth < 0:
        return False, depth, in_string, escape, False
    is_complete = (depth == 0 and last_non_whitespace == '}')
    return True, depth, in_string, escape, is_complete

def is_valid_json_prefix(text: str) -> bool:
    valid, _, _, _, _ = json_state(text)
    return valid

def extract_json(text: str):
    """Return the first complete JSON object found in text."""
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i+1]
    return None