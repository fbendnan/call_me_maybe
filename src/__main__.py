import argparse
import json
import sys
import os
from src.generator import LLMGenerator
from src.parser import FunctionDefinition, InputPrompt


def main() -> None:
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
            functions = json.load(f)
            if not isinstance(functions, list):
                raise ValueError("Functions definition file must contain a JSON array.")
            if not functions:
                raise ValueError("Functions definition file cannot be empty.")
        with open(args.input, 'r') as f:
            prompts = json.load(f)
            if not isinstance(prompts, list):
                raise ValueError("Functions calling test file must contain a JSON array.")
            if not prompts:
                raise ValueError("Functions calling test file cannot be empty.")
        for f in functions:
            _ = FunctionDefinition.model_validate(f)
        functions.append(unknowen_fun)

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error reading input files: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    generator = LLMGenerator(functions)
    for i, item in enumerate(prompts):
        user_prompt = ""
        try:
            print(f"\n========... start {i + 1} generation ...========")
            _ = InputPrompt.model_validate(item)
            user_prompt = item["prompt"]
            if user_prompt == "":
                raise ValueError('No prompt founded')
            result = generator.generate(user_prompt)
            results.append(result)
            print(f"==> [{i + 1}] Generated succesfully : ", result)
        except (KeyboardInterrupt, Exception) as e:
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
    print(f"\n==============[Output written to {args.output}]==============")


if __name__ == "__main__":
    main()
