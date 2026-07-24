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
                pos += 1
            
            elif part_token == "fun_name":
                func_names = [f['name'] for f in functions]
                
                allowed_tokens = set()
                for name in func_names:
                    for tid in model.encode(name).tolist()[0]:
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
                        break
                
                # if gen_name not in func_names:
                #     for name in func_names:
                #         if name.startswith(gen_name):
                #             remaining = name[len(gen_name):]
                #             if remaining:
                #                 for tid in model.encode(remaining).tolist()[0]:
                #                     logits = model.get_logits_from_input_ids(ids)
                #                     for token_id in range(len(logits)):
                #                         if token_id != tid:
                #                             logits[token_id] = float('-inf')
                #                     next_id = int(np.argmax(logits))
                #                     token = model.decode([next_id])
                #                     output += token
                #                     ids.append(next_id)
                #             break
                
                pos += 1
            
            elif part_token == "param_values":
                pos += 1
    
    return output


with open("data/input/function_calling_tests.json") as f:
    prompt = json.load(f)
    # print(prompt[0]['prompt'])

with open("data/input/functions_definition.json") as f:
    func = json.load(f)
    # print(func)

print(chat(prompt[8]['prompt'], func))