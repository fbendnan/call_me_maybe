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
        # next_id = int(np.argmax(logits))
        # print( ids)
        # print(logits)
        print(i)

        next_id = int(np.argmax(logits))
        if output == "":
            for token_id, logit in enumerate(logits):
                # logits = model.get_logits_from_input_ids(ids)
                # print(f"next_id = {next_id}")
                # print(np.argmax(logits))
                next_id = int(np.argmax(logits))
                token = model.decode([next_id])
                if token.startswith('{'):
                    print(token)
                    output += token
                    break
                logits[next_id] = float('-inf')


        token = model.decode([next_id])
        # print("token bera")
        if token in ["<|im_end|>", "</s>", "<|endoftext|>"]:
            break
        if output.startswith('{'):
            print(token)
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

print(chat(f"{prompt[2]['prompt']} and {func}, give me just a json output with prompt, "
            "and chose the name of function adapted from the list i give, parameters"))