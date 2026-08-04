import json
import numpy as np
from src.cache import get_tokens, model


class LLMGenerator:
    def __init__(self, functions, max_tokens=150):
        self.functions = functions
        self.max_tokens = max_tokens
        self.chosen_fun = None
        self.output = []
        self.ids = None
        self.step = 0
        self.do_comma = 1
        self.func_names = [f['name'] for f in functions]
        self.func_tokens = {name: get_tokens(name) for name in self.func_names}
        self.func_map = {f['name']: f for f in functions}

    def _generate_token(self, allowed_tokens):
        """Generate one token from the allowed set."""
        self.step += 1
        if self.step > self.max_tokens:
            raise RuntimeError("Max tokens exceeded")

        logits = model.get_logits_from_input_ids(self.ids)

        best_id = None
        best_score = float('-inf')
        for tok in allowed_tokens:
            if tok < len(logits) and logits[tok] > best_score:
                best_score = logits[tok]
                best_id = tok

        if best_id is None:
            raise RuntimeError("No valid token found")

        token = model.decode([best_id])
        self.output.append(token)
        self.ids.append(best_id)
        return token, best_id

    def _force_tokens(self, text):
        """Force a sequence of tokens."""
        for tid in get_tokens(text):
            self.ids.append(tid)
            self.output.append(model.decode([tid]))

    def _generate_function_name(self):
        """Generate function name using prefix matching."""
        generated = []
        possible = list(range(len(self.func_names)))

        while True:
            remaining = []
            for idx in possible:
                tokens = self.func_tokens[self.func_names[idx]]
                if len(tokens) >= len(generated) and \
                   tokens[:len(generated)] == generated:
                    remaining.append(idx)

            possible = remaining
            if len(possible) == 1:
                chosen = self.func_names[possible[0]]
                full = self.func_tokens[chosen]

                for tid in full[len(generated):]:
                    self.ids.append(tid)
                    self.output.append(model.decode([tid]))
                self.chosen_fun = self.func_map[chosen]
                return chosen

            next_tokens = set()
            for idx in possible:
                tokens = self.func_tokens[self.func_names[idx]]
                if len(tokens) > len(generated):
                    next_tokens.add(tokens[len(generated)])

            if not next_tokens:
                raise RuntimeError("No valid next token")

            _, tid = self._generate_token(next_tokens)
            generated.append(tid)

    def _generate_string_value(self):

        self._force_tokens('"')
        for _ in range(50):
            logits = model.get_logits_from_input_ids(self.ids)
            next_id = int(np.argmax(logits))
            token = model.decode([next_id])
            print(token)
            if '"' in token:
                if token.endswith('\\"'):
                    self.output.append(token)
                    self.ids.append(next_id)
                    continue
                elif token == '"':
                    self.output.append(token)
                    self.ids.append(next_id)
                    break
                elif token.endswith('"') and len(token) != 1:
                    self.output.append(token)
                    self.ids.append(next_id)
                    break
                elif token.endswith(','):
                    self.do_comma = 0
                    self.output.append(token)
                    self.ids.append(next_id)
                    break
                self._force_tokens('"')
                break
            self.output.append(token)
            self.ids.append(next_id)

    def _generate_number_value(self):
        """Generate a JSON number using constrained decoding."""
        res = ""
        for _ in range(20):
            logits = model.get_logits_from_input_ids(self.ids)
            next_id = int(np.argmax(logits))
            token = model.decode([next_id])
            if token in (",", "}", '}}'):
                return res
            for delim in (",", "}"):
                if delim in token:
                    return res
            res += token
            self.ids.append(next_id)

    def _generate_value(self, value_type):
        """Generate a value based on type."""
        if value_type.lower() == "string":
            self._generate_string_value()
        elif value_type.lower() == "number":
            res = self._generate_number_value()
            self.output += str(float(res))
        elif value_type.lower() == "integer":
            res = self._generate_number_value()
            self.output += str(int(res))

    def _generate_parameters(self):
        """Generate parameters object."""
        if self.chosen_fun is None:
            raise RuntimeError("No function chosen")

        self._force_tokens('{')
        params = self.chosen_fun['parameters']
        param_names = list(params.keys())

        for idx, pname in enumerate(param_names):
            self._force_tokens(f'"{pname}": ')
            param_type = params[pname].get('type', 'string')
            self._generate_value(param_type)
            if idx < len(param_names) - 1:
                if self.do_comma:
                    self._force_tokens(', ')
                self.do_comma = 1

        self._force_tokens('}')

    def generate(self, user_prompt):
        """Generate full JSON response."""
        self.output = []
        self.chosen_fun = None
        self.step = 0

        prompt = (
            f"You are a function-calling assistant. "
            f"Given a user request, output a JSON object "
            f"with exactly these keys: 'prompt', 'name', and 'parameters'.\n\n"
            f"Available functions (in JSON):\n{self.functions}\n\n"
            f"User request: {user_prompt}\n\n"
            f"Output JSON:"
        )
        self.ids = model.encode(prompt).tolist()[0]
        self._force_tokens('{"prompt": ')
        self._force_tokens(json.dumps(user_prompt))
        self._force_tokens(',"name": "')
        self._generate_function_name()
        self._force_tokens('", "parameters": ')
        self._generate_parameters()
        self._force_tokens('}')
        final_out = "".join(self.output)
        try:
            result = json.loads(final_out)
            return {
                "prompt": result['prompt'],
                "name": result["name"],
                "parameters": result["parameters"]
            }
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON: {final_out}") from e
