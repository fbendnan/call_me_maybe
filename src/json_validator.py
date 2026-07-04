import json

class ParseJson():
    def __init__(self):
        self.prompts = []
        self.fun_def = {}

    def check_bracets(self, js):
           ...

    def parse(self, file_name):
        with open(file_name, "r") as f:
            d = json.load(f)
            print(len(d))
            self.check_bracets(d)


p = ParseJson()
p.parse("data/input/function_calling_tests.json")