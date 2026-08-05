import json
import numpy as np
from src.cache import get_tokens_id, model
from typing import Any, Dict, List


class LLMGenerator:
    """Contain all functions needed for constrained decoding"""

    def __init__(self, functions: list[Dict[Any, Any]],
                 max_tokens: int = 150) -> None:
        self.functions = functions
        self.max_tokens = max_tokens
        self.chosen_fun: Dict[Any, Any] | None = None
        self.output: Any = []
        self.ids: list[int] = []
        self.step = 0
        self.do_comma = 1
        self.do_quote = 1
        self.func_names = [f["name"] for f in functions]
        self.func_tokens_id = {
            name: get_tokens_id(name) for name in self.func_names}
        self.func_map = {f["name"]: f for f in functions}

    def _generate_token_id(self, allowed_token_ids: Any) -> int:
        """Generate one token from the allowed set"""
        self.step += 1
        if self.step > self.max_tokens:
            raise RuntimeError("Max tokens exceeded")
        logits = np.array(model.get_logits_from_input_ids(self.ids))
        best_id: int | None = None
        allowed_ids = np.array(list(allowed_token_ids), dtype=np.int32)
        best_index = int(np.argmax(logits[allowed_ids]))
        best_id = allowed_ids[best_index]
        if best_id is None:
            raise RuntimeError("No valid token found")

        token = model.decode([best_id])
        self.output.append(token)
        self.ids.append(int(best_id))
        return best_id

    def _force_tokens(self, text: str) -> None:
        """Force a sequence of tokens"""
        tokens_ids = get_tokens_id(text)
        self.ids.extend(tokens_ids)
        self.output.extend(model.decode(tokens_ids))

    def _generate_function_name(self) -> None:
        """Generate function name using prefix matching"""
        generated: List[int] = []
        possible: List[int] = list(range(len(self.func_names)))
        while True:
            remaining: List[int] = []
            for idx in possible:
                tokens = self.func_tokens_id[self.func_names[idx]]
                if (
                    len(tokens) >= len(generated)
                    and tokens[: len(generated)] == generated
                ):
                    remaining.append(idx)

            possible = remaining
            if len(possible) == 1:
                chosen = self.func_names[possible[0]]
                full = self.func_tokens_id[chosen]
                remaining = full[len(generated):]
                
                self.ids.extend(remaining)
                self.output.append(model.decode(remaining))
                self.chosen_fun = self.func_map[chosen]
                return chosen
            next_tokens = set()
            for idx in possible:
                tokens = self.func_tokens_id[self.func_names[idx]]
                if len(tokens) > len(generated):
                    next_tokens.add(tokens[len(generated)])
            if not next_tokens:
                raise RuntimeError("No valid next token")

            tid = self._generate_token_id(next_tokens)
            generated.append(tid)

    def _generate_string_value(self) -> None:
        """Generate a String parameter using constrained decoding"""
        self._force_tokens('"')
        for _ in range(50):
            logits = model.get_logits_from_input_ids(self.ids)
            next_id = int(np.argmax(logits))
            token = model.decode([next_id])
            # print(token)
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
                    if token == '","':
                        self.do_quote = 0
                        self.do_comma = 0
                    self.output.append(token)
                    self.ids.append(next_id)
                    break
                elif token.endswith(","):
                    self.do_comma = 0
                    self.output.append(token)
                    self.ids.append(next_id)
                    break
                self._force_tokens('"')
                break
            self.output.append(token)
            self.ids.append(next_id)

    def _generate_number_value(self) -> str:
        """Generate a Number parameter using constrained decoding"""
        res = ""
        for _ in range(100):
            logits = model.get_logits_from_input_ids(self.ids)
            next_id = int(np.argmax(logits))
            token = model.decode([next_id])
            print(token)
            if any(d in token for d in (",", "}", "}}")):
                return res
            # if token in (",", "}", "}}"):
            #     return res
            # for delim in (",", "}"):
            #     if delim in token:
            #         return res
            res += token
            self.ids.append(next_id)
        return res

    def _generate_value(self, value_type: str) -> None:
        """Generate a value parameter based on type"""
        if value_type.lower() == "string":
            # print("param_names")
            self._generate_string_value()
        elif value_type.lower() == "number":
            res = self._generate_number_value()
            self.output += str(float(res))
        elif value_type.lower() == "integer":
            res = self._generate_number_value()
            self.output += str(int(res))

    def _generate_parameters(self) -> None:
        """Generate parameters object"""
        if self.chosen_fun is None:
            raise RuntimeError("No function chosen")

        self._force_tokens("{")
        params = self.chosen_fun["parameters"]
        param_names = list(params.keys())
        for idx, pname in enumerate(param_names):
            if self.do_quote:
                self._force_tokens(f'"{pname}": ')
            else:
                self._force_tokens(f'{pname}": ')
                self.do_quote = 1
            param_type = params[pname]["type"]
            self._generate_value(param_type)
            if idx < len(param_names) - 1:
                if self.do_comma:
                    self._force_tokens(", ")
                self.do_comma = 1

        self._force_tokens("}")

    def generate(self, user_prompt: str) -> Dict[Any, Any]:
        """Generate full JSON response"""
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
        self._force_tokens("}")
        final_out = "".join(self.output)
        try:
            result = json.loads(final_out)
            return {
                "prompt": result["prompt"],
                "name": result["name"],
                "parameters": result["parameters"],
            }
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON: {final_out}") from e
