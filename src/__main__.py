# from llm_sdk import Small_LLM_Model


# llm = Small_LLM_Model()
# print(llm.encode("hello bro"))

from llm_sdk import Small_LLM_Model
import numpy as np
import json

model = Small_LLM_Model()

def chat(prompt, max_tokens=100):
    ids = model.encode(prompt).tolist()[0]
    output = ""
    # open_brace_id = model.encode("{").tolist()[0][0]
    # quote_id = model.encode('"').tolist()[0][0]
    pos = 0
    my_json_structure = [
        ("forced", '{'),
        ("forced", ' '),
        ("forced", '"'),
        ("forced", "prompt"),
        ("forced", '"'),
        ("forced", ':'),
        ("not_forced", "prompt_value"),
        ("forced", ','),
        ("forced", ' '),
        ("forced", '"'),
        ("forced", "name"),
        ("forced", '"'),
        ("forced", ':'),
        ("not_forced", "fun_name"),
        ("forced", ','),
        ("forced", " "),
        ("forced", '"'),
        ("forced", "parameters"),
        ("forced", '"'),
        ("forced", ":"),
        ("forced", " "),
        ("forced", "{"),
        ("not_forced", "param_values"),
        ("forced", "}"),
        ("forced", "}"),
    ]

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)

        part_type, part_token = my_json_structure[pos]
        if pos < len(my_json_structure) and part_type == 'forced':
            token_pos = 0
            tok_id = model.encode(part_token).tolist()[0]
            len_tok = len(tok_id)
            while len_tok > token_pos:
                tok_now = tok_id[token_pos]
                for token_id in range(len(logits)):
                    if token_id != tok_now:
                        logits[token_id] = float('-inf')
                next_id = int(np.argmax(logits))
                token = model.decode([next_id])
                output += token
                token_pos += 1
                ids.append(next_id)
            pos += 1
            if pos == len(my_json_structure):
                break

        elif pos < len(my_json_structure) and part_type == 'not_forced':
            if part_token == "prompt_value":
                ...
            pos += 1
            
            for _ in range(10):
                next_id = int(np.argmax(logits))
                token = model.decode([next_id])
                output += token
                ids.append(next_id)
                logits = model.get_logits_from_input_ids(ids)
        if token in ["<|im_end|>", "</s>", "<|endoftext|>"]:
            break



            # elif part_token == "fun_name":
            #     ...
            # elif part_token == "param_values":
            #     ...


        # next_id = int(np.argmax(logits))
        # token = model.decode([next_id])
        # # print("token bera")
        # if token in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        #     break
        #     # print(token)
        # output += token
        
        # pos += 1
        # if pos == len(my_json_structure):
        #     break
        # ids.append(next_id)
    
    return output


with open("data/input/function_calling_tests.json") as f:
    prompt = json.load(f)
    # print(prompt[0]['prompt'])

with open("data/input/functions_definition.json") as f:
    func = json.load(f)
    # print(func)

print(chat(f"prompt : {prompt[9]['prompt']}, functions: {func},"
            "based on the prompt chose the right function to it and give me just a JSON "
            "output (no extra text) with informations like prompt, name (of function used), parameters"))