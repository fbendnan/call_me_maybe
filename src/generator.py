import numpy as np
from src.cache import get_tokens, SPECIAL_IDS, model

class LLMGenerator:
    def __init__(self, user_prompt, functions, max_tokens=80):
        self.user_prompt = user_prompt
        self.functions = functions
        self.max_tokens = max_tokens
        self.chosen_fun = None
        self.output = ""
        self.ids = None

    def _force_tokens(self, text):
        for tid in get_tokens(text):
            logits = model.get_logits_from_input_ids(self.ids)
            mask = np.full(len(logits), -float('inf'), dtype=np.float32)
            mask[tid] = 0.0
            logits = np.array(logits, dtype=np.float32) + mask
            next_id = int(np.argmax(logits))
            token = model.decode([next_id])
            self.output += token
            self.ids.append(next_id)

    def _generate_value(self):
        comma_id = SPECIAL_IDS['comma']
        brace_id = SPECIAL_IDS['close_brace']
        for _ in range(20):
            logits = model.get_logits_from_input_ids(self.ids)
            next_id = int(np.argmax(logits))
            if next_id == comma_id or next_id == brace_id:
                break
            token = model.decode([next_id])
            self.output += token
            self.ids.append(next_id)

    def _generate_function_name(self):
        func_names = [f['name'] for f in self.functions]
        
        allowed_tokens = set()
        for name in func_names:
            for tid in get_tokens(name):
                allowed_tokens.add(tid)
        
        gen_name = ""
        for _ in range(10):
            logits = model.get_logits_from_input_ids(self.ids)
            
            mask = np.full(len(logits), -float('inf'), dtype=np.float32)
            for tid in allowed_tokens:
                mask[tid] = 0.0
            logits = np.array(logits, dtype=np.float32) + mask
            #####check the third func sherk and the ouput steps
            next_id = int(np.argmax(logits))
            if next_id not in allowed_tokens:
                break
            token = model.decode([next_id])
            gen_name += token
            print(f"fun name = {gen_name}")
            self.output += token
            self.ids.append(next_id)
            
            if gen_name in func_names:
                self.chosen_fun = next(f for f in self.functions if f['name'] == gen_name)
                break
        
        if self.chosen_fun is None:
            for name in func_names:
                if name.startswith(gen_name):
                    remaining = name[len(gen_name):]
                    if remaining:
                        self._force_tokens(remaining)
                    self.chosen_fun = next(f for f in self.functions if f['name'] == name)
                    break

    def _generate_parameters(self):
        if self.chosen_fun is None:
            param_names = []
        else:
            params = self.chosen_fun['parameters']
            param_names = list(params.keys())

        for idx, pname in enumerate(param_names):
            key_str = f'"{pname}": '
            self._force_tokens(key_str)
            self._generate_value()
            if idx < len(param_names) - 1:
                self._force_tokens(", ")

    def generate(self):
        full_prompt = f"""User request: {self.user_prompt}
                Available functions: {', '.join([f['name'] for f in self.functions])}
                Return ONLY JSON with prompt, name, and parameters."""

        self.ids = model.encode(full_prompt).tolist()[0]
        self.output = ""
        pos = 0

        skeleton = [
            ("forced", '{'),
            ("forced", ' '),
            ("forced", '"'),
            ("forced", "prompt"),
            ("forced", '"'),
            ("forced", ':'),
            ("forced", ' '),
            ("forced", '"'),
            ("not_forced", "prompt_value"),
            ("forced", '"'),
            ("forced", ','),
            ("forced", ' '),
            ("forced", '"'),
            ("forced", "name"),
            ("forced", '"'),
            ("forced", ':'),
            ("forced", ' '),
            ("forced", '"'),
            ("not_forced", "fun_name"),
            ("forced", '"'),
            ("forced", ','),
            ("forced", ' '),
            ("forced", '"'),
            ("forced", "parameters"),
            ("forced", '"'),
            ("forced", ':'),
            ("forced", ' '),
            ("forced", '{'),
            ("not_forced", "param_values"),
            ("forced", '}'),
            ("forced", '}'),
        ]

        for step in range(self.max_tokens):
            if pos >= len(skeleton):
                break

            part_type, part_token = skeleton[pos]

            if part_type == 'forced':
                self._force_tokens(part_token)
                pos += 1
                if pos == len(skeleton):
                    break

            else:
                if part_token == "prompt_value":
                    self._force_tokens(self.user_prompt)
                elif part_token == "fun_name":
                    self._generate_function_name()
                elif part_token == "param_values":
                    self._generate_parameters()
                pos += 1
            print(self.output)
        return self.output