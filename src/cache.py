from llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()

_token_cache = {}

def get_tokens(text: str) -> list[int]:
    if text not in _token_cache:
        _token_cache[text] = model.encode(text).tolist()[0]
    return _token_cache[text]

def _get_special_ids():
    return {
        'quote': get_tokens('"')[0],
        'comma': get_tokens(',')[0],
        'brace_close': get_tokens('}')[0],
        'brace_open': get_tokens('{')[0],
        'colon': get_tokens(':')[0],
        'space': get_tokens(' ')[0],
        'bracket_open': get_tokens('[')[0],
        'bracket_close': get_tokens(']')[0],
    }

SPECIAL_IDS = _get_special_ids()