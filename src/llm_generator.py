from llm_sdk import Small_LLM_Model
import numpy as np
import json

model = Small_LLM_Model()

class llm_generator:
    def __init__(self, user_prompt, functions, max_tokens=200):
        self.user_prompt = user_prompt
        self.functions = functions
        self.max_tokens = max_tokens
        self.chosen_fun = None
        self.output = ""
        self.ids = None

    def gen_forced_part(self, part_token):
        tok_ids = model.encode(part_token).tolist()[0]
        for tok_id in tok_ids:
            logits = model.get_logits_from_input_ids(self.ids)
            for token_id in range(len(logits)):
                if token_id != tok_id:
                    logits[token_id] = float('-inf')
            next_id = int(np.argmax(logits))
            token = model.decode([next_id])
            self.output += token
            self.ids.append(next_id)

    def gen_prompt(self):
        prompt_tokens = model.encode(self.user_prompt).tolist()[0]
        for tok_id in prompt_tokens:
            logits = model.get_logits_from_input_ids(self.ids)
            for token_id in range(len(logits)):
                if token_id != tok_id:
                    logits[token_id] = float('-inf')
            next_id = int(np.argmax(logits))
            token = model.decode([next_id])
            self.output += token
            self.ids.append(next_id)

    def generate_fun_name(self):
        func_names = [f['name'] for f in self.functions]
        allowed_tokens = set()
        for name in func_names:
            func_name_tokens = model.encode(name).tolist()[0]
            for tid in func_name_tokens:
                allowed_tokens.add(tid)
        
        gen_name = ""
        for _ in range(30):
            logits = model.get_logits_from_input_ids(self.ids)
            for token_id in range(len(logits)):
                if token_id not in allowed_tokens:
                    logits[token_id] = float('-inf')
            
            next_id = int(np.argmax(logits))
            token = model.decode([next_id])
            
            if next_id not in allowed_tokens:
                break
            
            gen_name += token
            self.output += token
            self.ids.append(next_id)
            
            if gen_name in func_names:
                self.chosen_fun = next(func for func in self.functions if func['name'] == gen_name)
                break
        
        if gen_name not in func_names:
            ...

    def gen_param_values(self):
        if self.chosen_fun is None:
            self.chosen_fun = functions[0]
        params = self.chosen_fun['parameters']
        param_names = list(params.keys())

        for index, param_name in enumerate(param_names):
            param_name = f'"{param_name}": '
            param_name_tokens = model.encode(param_name).tolist()[0]
            for tok_id in param_name_tokens:
                logits = model.get_logits_from_input_ids(self.ids)
                for token_id in range(len(logits)):
                    if tok_id != token_id:
                        logits[token_id] = float('-inf')
                next_id = int(np.argmax(logits))
                token = model.decode([next_id])
                self.output += token
                self.ids.append(next_id)

            value = ""
            for _ in range(15):
                logits = model.get_logits_from_input_ids(self.ids)
                next_id = int(np.argmax(logits))
                token = model.decode([next_id])
                if '}' in token or ',' in token:
                    break
                value += token
                self.output += token
                self.ids.append(next_id)

            if index < len(param_names) - 1:
                virg_token = model.encode(', ').tolist()[0]
                for tok_id in virg_token:
                    logits = model.get_logits_from_input_ids(self.ids)
                    for token_id in range(len(logits)):
                        if token_id != tok_id:
                            logits[token_id] = float('-inf')
                    next_id = int(np.argmax(logits))
                    token = model.decode([next_id])
                    self.output+= token
                    self.ids.append(next_id)

    def chat(self):     
        full_prompt = f'''User request: {self.user_prompt}
        Available functions: {', '.join([f['name'] for f in self.functions])}
        Return ONLY JSON with prompt, name, and parameters.'''

        self.ids = model.encode(full_prompt).tolist()[0]
        pos = 0
        self.output = ""    
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

            logits = model.get_logits_from_input_ids(self.ids)
            part_type, part_token = skeleton[pos]
            if part_type == 'forced':
                self.gen_forced_part(part_token)
                pos += 1

            elif part_type == 'not_forced':
                if part_token == "prompt_value":
                    self.gen_prompt()

                elif part_token == "fun_name":
                    self.generate_fun_name()

                elif part_token == "param_values":
                    self.gen_param_values()
                pos += 1
        
        return self.output