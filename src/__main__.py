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
    open_brace_id = model.encode("{").tolist()[0][0]
    quote_id = model.encode('"').tolist()[0][0]
    
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)

        if output == "":
            for token_id in range(len(logits)):
                if token_id != open_brace_id:
                    logits[token_id] = float('-inf')

        elif output == "{":
            for token_id in range(len(logits)):
                    if token_id != quote_id:
                        logits[token_id] = float('-inf')

        next_id = int(np.argmax(logits))
        token = model.decode([next_id])
        # print("token bera")
        if token in ["<|im_end|>", "</s>", "<|endoftext|>"]:
            break
            # print(token)
        output += token
        ids.append(next_id)
    
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