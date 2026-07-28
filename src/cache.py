from llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()

_token_cache = {}

def get_tokens(text: str) -> list[int]:
    if text not in _token_cache:
        _token_cache[text] = model.encode(text).tolist()[0]
    return _token_cache[text]

# Global vocabulary cache - loaded once
_vocab = None
_id_to_token = None
_vocab_size = None
_token_strings = None  # Global cache for token strings

def get_vocab():
    global _vocab, _id_to_token, _vocab_size, _token_strings
    if _vocab is None:
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, 'r') as f:
            raw_vocab = json.load(f)
        _vocab = {int(v): k for k, v in raw_vocab.items()}
        _id_to_token = _vocab
        _vocab_size = len(_vocab)
        # Cache all token strings for instant access
        _token_strings = {i: _vocab.get(i, '') for i in range(_vocab_size)}
    return _id_to_token

def get_token_string(token_id: int) -> str:
    get_vocab()
    return _token_strings.get(token_id, '')

def get_vocab_size():
    get_vocab()
    return _vocab_size

valid_string_tokens = None
valid_number_continuation_tokens = None

def get_valid_string_tokens():
    """Get or compute valid string tokens (without quote)."""
    global valid_string_tokens
    if valid_string_tokens is not None:
        return valid_string_tokens
    
    vocab_size = get_vocab_size()
    quote_id = get_quote_id()

    content_tokens = set()
    with_quote = set()
    
    for tid in range(vocab_size):
        token_str = _token_strings.get(tid, '')
        if not token_str:
            continue
        
        # Skip the quote token itself for content
        if tid == quote_id:
            with_quote.add(tid)
            continue
        
        # Check if valid inside JSON string
        is_valid = True
        for char in token_str:
            if char == '"':
                is_valid = False
                break
            if ord(char) < 32 and char not in ('\n', '\r', '\t'):
                is_valid = False
                break
        
        if is_valid:
            content_tokens.add(tid)
            with_quote.add(tid)
    
    # Add quote to with_quote set
    with_quote.add(quote_id)
    
    _valid_string_content_tokens = content_tokens
    valid_string_tokens = with_quote
    return with_quote

def get_valid_number_tokens():
    """Get or compute valid number tokens."""
    global valid_number_continuation_tokens
    if valid_number_continuation_tokens is not None:
        return valid_number_continuation_tokens
    
    vocab_size = get_vocab_size()
    valid_tokens = set()
    
    for tid in range(vocab_size):
        token_str = _token_strings.get(tid, '')
        if not token_str:
            continue
        
        # Only tokens with number characters
        if all(c in '0123456789.-eE+' for c in token_str):
            # Must contain at least one digit
            if any(c.isdigit() for c in token_str):
                valid_tokens.add(tid)
    
    valid_number_continuation_tokens = valid_tokens
    return valid_tokens

# Pre-compute special IDs
_quote_id = None
_comma_id = None
_brace_close_id = None
_brace_open_id = None
_colon_id = None
_space_id = None
_bracket_open_id = None
_bracket_close_id = None

def _init_special_ids():
    global _quote_id, _comma_id, _brace_close_id, _brace_open_id, _colon_id, _space_id
    global _bracket_open_id, _bracket_close_id
    if _quote_id is not None:
        return
    _quote_id = get_tokens('"')[0]
    _comma_id = get_tokens(',')[0]
    _brace_close_id = get_tokens('}')[0]
    _brace_open_id = get_tokens('{')[0]
    _colon_id = get_tokens(':')[0]
    _space_id = get_tokens(' ')[0]
    _bracket_open_id = get_tokens('[')[0]
    _bracket_close_id = get_tokens(']')[0]

def get_quote_id():
    _init_special_ids()
    return _quote_id

def get_comma_id():
    _init_special_ids()
    return _comma_id

def get_brace_close_id():
    _init_special_ids()
    return _brace_close_id

def get_brace_open_id():
    _init_special_ids()
    return _brace_open_id

def get_colon_id():
    _init_special_ids()
    return _colon_id

def get_space_id():
    _init_special_ids()
    return _space_id

def get_bracket_open_id():
    _init_special_ids()
    return _bracket_open_id

def get_bracket_close_id():
    _init_special_ids()
    return _bracket_close_id