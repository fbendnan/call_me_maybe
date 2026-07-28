import json
from src.cache import (
    get_tokens, get_token_string, model,
    get_valid_string_tokens, get_valid_number_tokens,
    get_quote_id, get_comma_id, get_brace_close_id,
    get_brace_open_id, get_colon_id, get_bracket_open_id,
    get_bracket_close_id
)

class LLMGenerator:
    def __init__(self, functions, max_tokens=300):
        self.functions = functions
        self.max_tokens = max_tokens
        self.chosen_fun = None
        self.output = ""
        self.ids = None
        self.step = 0
        self.function_map = {f["name"]: f for f in functions}
        self.function_names = list(self.function_map.keys())
        self.function_tokens = {}
        for name in self.function_names:
            self.function_tokens[name] = get_tokens(name)

        self.all_string_tokens = get_valid_string_tokens()
        self.number_tokens = get_valid_number_tokens()

        self.number_token_strings = {
            tok: get_token_string(tok)
            for tok in self.number_tokens
        }

        self.quote_id = get_quote_id()
        self.comma_id = get_comma_id()
        self.brace_close_id = get_brace_close_id()
        self.brace_open_id = get_brace_open_id()
        self.colon_id = get_colon_id()
        self.bracket_open_id = get_bracket_open_id()
        self.bracket_close_id = get_bracket_close_id()
        self.delimiters = {self.comma_id, self.brace_close_id, self.bracket_close_id}

        self.space_colon_tokens = get_tokens(': ')
        self.ids = None
        

    def _force_token_ids(self, token_ids):
        """Force a sequence of token IDs using constrained decoding."""
        for tok in token_ids:
            self._generate_token_with_constraints({tok})

    def _force_tokens(self, text):
        """Force a specific sequence of tokens using constrained decoding."""
        for tok in get_tokens(text):
            self._generate_token_with_constraints({tok})

    def _generate_token_with_constraints(self, allowed_tokens):
        """
        Generate one token under constraints.
        No mask allocation. No numpy copy. Just a simple loop.
        """
        self.step += 1
        if self.step > self.max_tokens:
            raise RuntimeError("Max tokens exceeded")

        logits = model.get_logits_from_input_ids(self.ids)

        best_id = None
        best_score = float("-inf")

        for tok in allowed_tokens:
            if 0 <= tok < len(logits):
                score = logits[tok]
                if score > best_score:
                    best_score = score
                    best_id = tok

        if best_id is None:
            raise RuntimeError("No valid token found")

        token = model.decode([best_id])
        print("from _generate_token_with_constraints", token)
        self.output += token
        self.ids.append(best_id)

        return token, best_id

    def _generate_function_name(self):
        """Generate function name using prefix matching with constrained decoding."""
        generated_tokens = []

        possible_funcs = list(range(len(self.function_names)))
        while True:
            remaining_funcs = []
            for idx in possible_funcs:
                func_tokens = self.function_tokens[self.function_names[idx]]
                if len(func_tokens) >= len(generated_tokens):
                    if func_tokens[:len(generated_tokens)] == generated_tokens:
                        remaining_funcs.append(idx)

            possible_funcs = remaining_funcs
            
            if not possible_funcs:
                raise RuntimeError("No valid function name continuation found.")
            if len(possible_funcs) == 1:
                chosen_fun_idx = possible_funcs[0]
                chosen_fun_name = self.function_names[chosen_fun_idx]
                full_tokens = self.function_tokens[chosen_fun_name]
                remaining_tokens = full_tokens[len(generated_tokens):]

                for tok in remaining_tokens:
                    _, _ = self._generate_token_with_constraints({tok})
                
                self.chosen_fun = self.function_map[chosen_fun_name]
                return chosen_fun_name

            allowed_next_tokens = set()
            for idx in possible_funcs:
                func_tokens = self.function_tokens[self.function_names[idx]]
                # print(f"func_ token = ", func_tokens)
                if len(func_tokens) > len(generated_tokens):
                    allowed_next_tokens.add(func_tokens[len(generated_tokens)])
                    # print(f"in index {idx} :", allowed_next_tokens)
            
            if not allowed_next_tokens:
                raise RuntimeError("No allowed next tokens.")
            
            _, next_id = self._generate_token_with_constraints(allowed_next_tokens)
            generated_tokens.append(next_id)

    def _generate_string_value(self):
        """Generate a JSON string value with proper constraints."""
        self._force_tokens('"')

        while True:
            _, next_id = self._generate_token_with_constraints(self.all_string_tokens)
            if next_id == self.quote_id:
            # tok = model.decode([next_id])
            # if '"' in tok or '}' in tok:
                break

    def _generate_number_value(self):
        """Generate a JSON number value with proper constraints."""
        current_number = ""

        is_valid_number = False

        for _ in range(20):
            if is_valid_number:
                raw_logits = model.get_logits_from_input_ids(self.ids)
                top_id = int(max(range(len(raw_logits)), key=lambda i: raw_logits[i]))
                tok = model.decode([top_id])
                # if top_id in self.delimiters:
                if "," in tok or "}" in tok:
                    return

            valid_tokens = []
            for tok, token_str in self.number_token_strings.items():

                if any(c in token_str for c in (' ', '"', '{', '}', '[', ']')):
                    continue
                
                test_str = current_number + token_str
                if self._is_valid_number_prefix(test_str):
                    valid_tokens.append(tok)
            

            if not valid_tokens:
                break
            
            token_str, _ = self._generate_token_with_constraints(valid_tokens)
            current_number += token_str

            try:
                float(current_number)
                is_valid_number = True

                raw_logits_after = model.get_logits_from_input_ids(self.ids)
                top_id_after = int(max(range(len(raw_logits_after)), key=lambda i: raw_logits_after[i]))
                if top_id_after in self.delimiters:
                    return
            except ValueError:
                is_valid_number = False

    def _is_valid_number_prefix(self, test_str):
        """Check if a string is a valid JSON number prefix."""
        if not test_str:
            return False
        if '..' in test_str or '.-' in test_str or '-.' in test_str:
            return False
        if test_str == '.' or test_str == '-' or test_str == '-.':
            return False
        
        try:
            float(test_str)
            return True
        except ValueError:

            if test_str.endswith('.') or test_str.endswith('e') or test_str.endswith('E'):
                test_clean = test_str.rstrip('.eE')
                if test_clean.replace('-', '').replace('+', '').isdigit():
                    return True
            if test_str.endswith('+') or test_str.endswith('-'):
                test_clean = test_str.rstrip('+-')
                if test_clean and test_clean.replace('.', '').replace('e', '').replace('E', '').isdigit():
                    return True
            return False

    def _generate_value(self, value_type):
        """Generate a value based on the full schema."""
        # if isinstance(schema, str):
        #     schema = {"type": schema}
                
        if value_type == "string":
            self._generate_string_value()
        elif value_type in ["number", "integer"]:
            self._generate_number_value()
        elif value_type == "boolean":
            self._generate_boolean_value()

    def _generate_parameters(self):
        """Generate the parameters object."""
        if self.chosen_fun is None:
            raise RuntimeError("No function chosen.")
        
        self._force_tokens('{')
        
        params = self.chosen_fun["parameters"]
        param_names = list(params.keys())
        
        for idx, param_name in enumerate(param_names):
            self._force_tokens(f'"{param_name}": ')
            param_type = params[param_name].get("type", "string")
            self._generate_value(param_type)
            if idx < len(param_names) - 1:
                self._force_tokens(', ')
        
        self._force_tokens('}')

    def generate(self, user_prompt):
        """Generate the full JSON response."""
        self.output = ""
        self.chosen_fun = None
        user_prompt_j = json.dumps(user_prompt)
        user_prompt_tok= get_tokens(user_prompt_j)
        full_prompt = (
                    f"You are a function-calling assistant. Given a user request, output a JSON object "
                    f"with exactly these keys: 'prompt', 'name', and 'parameters'.\n\n"
                    f"Available functions (in JSON):\n{self.functions}\n\n"
                    f"User request: {user_prompt}\n\n"
                    f"Output JSON:"
                )
        self.ids = model.encode(full_prompt).tolist()[0]
        # self._force_token_ids(get_tokens('{"prompt": '))
        # self._force_token_ids(user_prompt_tok)
        # self._force_token_ids(get_tokens(', "name": "'))
        self._generate_function_name()
        self._force_token_ids(get_tokens('", "parameters": '))
        self._generate_parameters()
        self._force_tokens('}')
        
        try:
            result = json.loads(self.output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Generated invalid JSON: {self.output}") from e
        
        return {
            "prompt": user_prompt,
            "name": result["name"],
            "parameters": result["parameters"]
        }