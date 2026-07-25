# from llm_sdk import Small_LLM_Model


# llm = Small_LLM_Model()
# print(llm.encode("hello bro"))
import json
from src.llm_generator import llm_generator


with open("data/input/function_calling_tests.json") as f:
    prompt = json.load(f)
    # print(prompt[0]['prompt'])

with open("data/input/functions_definition.json") as f:
    func = json.load(f)
    # print(func)

ll = llm_generator(prompt[9]['prompt'], func)
print(ll.chat())

# result = []
# for p in prompt:
#     print("yyy")
#     res = chat(p['prompt'], func)
#     result.append(res)

# with open("data/output.json", "w") as file:
#     for res in result:
#         file.write(res)