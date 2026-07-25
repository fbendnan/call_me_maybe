from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

_token_cache = {}

def get_tokens(text: str) -> list[int]:
    if text not in _token_cache:
        _token_cache[text] = model.encode(text).tolist()[0]
    return _token_cache[text]

def _get_special_ids():
    return {
        'comma': get_tokens(',')[0],
        'close_brace': get_tokens('}')[0],
        'open_brace': get_tokens('{')[0],
        'colon': get_tokens(':')[0],
        'space': get_tokens(' ')[0],
        'quote': get_tokens('"')[0],
    }

SPECIAL_IDS = _get_special_ids()