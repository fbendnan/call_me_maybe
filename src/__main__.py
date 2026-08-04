import argparse
import json
import sys
import os
from src.generator import LLMGenerator
from src.parser import FunctionDefinition, InputPrompt


def main():
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding.")
    parser.add_argument('--functions_definition',
                        default='data/input/functions_definition.json',
                        help='Path to function definitions JSON file')
    parser.add_argument('--input',
                        default='data/input/function_calling_tests.json',
                        help='Path to input prompts JSON file')
    parser.add_argument('--output',
                        default='data/output/function_calls.json',
                        help='Path to output JSON file')
    args = parser.parse_args()

    unknowen_fun = {
        "name": "No function available for that prompt",
        "description": ("Select this function only if no available function"
                        " matches the user's prompt after considering all "
                        "provided function definitions."),
        "parameters": {},
        "returns": {}
    }

    try:
        with open(args.functions_definition, 'r') as f:
            functions: FunctionDefinition = json.load(f)
            functions.append(unknowen_fun)
        with open(args.input, 'r') as f:
            prompts: InputPrompt = json.load(f)
        for f in functions:
            _ = FunctionDefinition.model_validate(f)
    except Exception as e:
        print(f"Error reading input files: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    generator = LLMGenerator(functions)
    for i, item in enumerate(prompts):
        user_prompt = ""
        try:
            _ = InputPrompt.model_validate(item)
            user_prompt = item["prompt"]
            if user_prompt == "":
                raise ValueError('No prompt founded')
            result = generator.generate(user_prompt)
            results.append(result)
            print(f"Generated {i + 1} : ", result)
        except Exception as e:
            print(f"Error generating for prompt '{user_prompt}': {e}",
                  file=sys.stderr)
            results.append({
                "prompt": user_prompt,
                "name": "",
                "parameters": {}
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
