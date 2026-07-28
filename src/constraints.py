from typing import Set, List
from src.cache import (
    get_tokens,
    load_vocab,
    get_quote_token,
)
import re

def allowed_for_number(current_number: str) -> Set[int]:
    """
    Return token IDs that can be appended to `current_number` to keep it a valid JSON number prefix.
    """
    vocab = load_vocab()
    pattern = re.compile(r'^-?(0|[1-9]\d*)?(\.\d*)?([eE][+-]?\d*)?$')
    allowed = set()
    for tid, token_str in vocab.items():
        if any(c in token_str for c in (',', '}', '"', ' ', '\n', '\t')):
            continue
        if pattern.match(current_number + token_str):
            allowed.add(tid)
    return allowed

def allowed_for_string_inside() -> Set[int]:
    """
    Return token IDs that can appear inside a JSON string (not including opening/closing quote).
    Allows the closing quote token as well.
    """
    vocab = load_vocab()
    quote_tok = get_quote_token()
    allowed = set()
    for tid, token_str in vocab.items():
        if tid == quote_tok:
            allowed.add(tid)
        else:
            if '"' in token_str or any(ord(c) < 32 for c in token_str):
                continue
            allowed.add(tid)
    return allowed

def allowed_for_boolean() -> Set[int]:
    return set(get_tokens('true') + get_tokens('false'))

def allowed_for_null() -> Set[int]:
    return set(get_tokens('null'))

def build_name_trie(function_names: List[str]) -> dict:
    """
    Build a trie from token sequences of function names.
    Root node: {'children': {}, 'terminal': False}
    """
    trie = {'children': {}, 'terminal': False}
    for name in function_names:
        tokens = get_tokens(name)
        node = trie
        for tid in tokens:
            if tid not in node['children']:
                node['children'][tid] = {'children': {}, 'terminal': False}
            node = node['children'][tid]
        node['terminal'] = True
    return trie