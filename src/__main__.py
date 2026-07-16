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
    

    
    for i in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        
        next_id = int(np.argmax(logits))

        if output != "" and output.startswith('{') and len(output) == 1:
            for token_id, logit in enumerate(logits):
                next_id = int(np.argmax(logits))
                token = model.decode([next_id])
                if token.startswith('"') or token.strip() == '"':
                    # print(token)
                    output += token
                    break
                logits[next_id] = float('-inf')

        if output == "":
            for token_id, logit in enumerate(logits):
                next_id = int(np.argmax(logits))
                token = model.decode([next_id])
                if token == '{':
                    # print(token)
                    output += token
                    break
                logits[next_id] = float('-inf')
        
            



        token = model.decode([next_id])
        # print("token bera")
        if token in ["<|im_end|>", "</s>", "<|endoftext|>"]:
            break
            # print(token)
        output += token
        ids.append(next_id)
        
        if len(output) > 200:
            break
    
    return output


with open("data/input/function_calling_tests.json") as f:
    prompt = json.load(f)
    # print(prompt[0]['prompt'])

with open("data/input/functions_definition.json") as f:
    func = json.load(f)
    # print(func)

print(chat(f"prompt : {prompt[7]['prompt']}, functions: {func},"
            "based on the prompt chose the right function to it and give me just a JSON "
            "output (no extra text) with informations like prompt, name (of function used), parameters"))