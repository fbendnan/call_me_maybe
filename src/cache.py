from llm_sdk.llm_sdk.llm_sdk import Small_LLM_Model


model = Small_LLM_Model()
_token_cache = {}


def get_tokens(text: str) -> list[int]:
    if text not in _token_cache:
        _token_cache[text] = model.encode(text).tolist()[0]
    return _token_cache[text]
