# from llm_sdk import Small_LLM_Model


# llm = Small_LLM_Model()
# print(llm.encode("hello bro"))
from llm_sdk import Small_LLM_Model
import numpy as np
import json

model = Small_LLM_Model()

def chat(user_prompt, functions, max_tokens=200):
    
    prompt = f'''User request: {user_prompt}
Available functions: {', '.join([f['name'] for f in functions])}
Return ONLY JSON with prompt, name, and parameters.'''

    ids = model.encode(prompt).tolist()[0]
    output = ""
    pos = 0
    chosen_fun = None
    
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

    for step in range(max_tokens):
        if pos >= len(skeleton):
            break
            
        logits = model.get_logits_from_input_ids(ids)
        part_type, part_token = skeleton[pos]

        if part_type == 'forced':
            tok_ids = model.encode(part_token).tolist()[0]
            for tok_id in tok_ids:
                logits = model.get_logits_from_input_ids(ids)
                for token_id in range(len(logits)):
                    if token_id != tok_id:
                        logits[token_id] = float('-inf')
                next_id = int(np.argmax(logits))
                token = model.decode([next_id])
                output += token
                ids.append(next_id)
            pos += 1

        elif part_type == 'not_forced':
            if part_token == "prompt_value":
                prompt_tokens = model.encode(user_prompt).tolist()[0]
                for tok_id in prompt_tokens:
                    logits = model.get_logits_from_input_ids(ids)
                    for token_id in range(len(logits)):
                        if token_id != tok_id:
                            logits[token_id] = float('-inf')
                    next_id = int(np.argmax(logits))
                    token = model.decode([next_id])
                    output += token
                    ids.append(next_id)
            elif part_token == "fun_name":
                func_names = [f['name'] for f in functions]
                
                allowed_tokens = set()
                for name in func_names:
                    func_name_tokens = model.encode(name).tolist()[0]
                    for tid in func_name_tokens:
                        allowed_tokens.add(tid)
                
                gen_name = ""
                for _ in range(30):
                    logits = model.get_logits_from_input_ids(ids)
                    for token_id in range(len(logits)):
                        if token_id not in allowed_tokens:
                            logits[token_id] = float('-inf')
                    
                    next_id = int(np.argmax(logits))
                    token = model.decode([next_id])
                    
                    if next_id not in allowed_tokens:
                        break
                    
                    gen_name += token
                    output += token
                    ids.append(next_id)
                    
                    if gen_name in func_names:
                        chosen_fun = next(func for func in functions if func['name'] == gen_name)
                        break

                print(gen_name)
            elif part_token == "param_values":
                if chosen_fun is None:
                    chosen_fun = functions[0]
                params = chosen_fun['parameters']
                param_names = list(params.keys())

                for index, param_name in enumerate(param_names):
                    param_name = f'"{param_name}": '
                    param_name_tokens = model.encode(param_name).tolist()[0]
                    for tok_id in param_name_tokens:
                        logits = model.get_logits_from_input_ids(ids)
                        for token_id in range(len(logits)):
                            if tok_id != token_id:
                                logits[token_id] = float('-inf')
                        next_id = int(np.argmax(logits))
                        token = model.decode([next_id])
                        output += token
                        ids.append(next_id)

                    value = ""
                    for _ in range(15):
                        logits = model.get_logits_from_input_ids(ids)
                        next_id = int(np.argmax(logits))
                        token = model.decode([next_id])
                        if '}' in token or ',' in token:
                            break
                        value += token
                        output += token
                        ids.append(next_id)

                    if index < len(param_names) - 1:
                        virg_token = model.encode(', ').tolist()[0]
                        for tok_id in virg_token:
                            logits = model.get_logits_from_input_ids(ids)
                            for token_id in range(len(logits)):
                                if token_id != tok_id:
                                    logits[token_id] = float('-inf')
                            next_id = int(np.argmax(logits))
                            token = model.decode([next_id])
                            output+= token
                            ids.append(next_id)
            pos += 1
    return output


with open("data/input/function_calling_tests.json") as f:
    prompt = json.load(f)
    # print(prompt[0]['prompt'])

with open("data/input/functions_definition.json") as f:
    func = json.load(f)
    # print(func)

result = []
for p in prompt:
    res = chat(p['prompt'], func)
    print(res)
    result.append(res)

with open("data/output.json", "w") as file:
    for res in result:
        file.write(res)