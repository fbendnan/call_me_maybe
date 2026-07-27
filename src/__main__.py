import argparse
import json
import sys
import os
from src.models import validate_output
from src.generator import LLMGenerator
from src.validator import extract_json

def main():
    parser = argparse.ArgumentParser(description="Function calling with constrained decoding.")
    parser.add_argument('--functions_definition', default='data/input/functions_definition.json',
                        help='Path to function definitions JSON file')
    parser.add_argument('--input', default='data/input/function_calling_tests.json',
                        help='Path to input prompts JSON file')
    parser.add_argument('--output', default='data/output/function_calls.json',
                        help='Path to output JSON file')
    args = parser.parse_args()

    try:
        with open(args.functions_definition, 'r') as f:
            functions = json.load(f)
        with open(args.input, 'r') as f:
            prompts = json.load(f)
    except Exception as e:
        print(f"Error reading input files: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    for item in prompts:
        user_prompt = item.get('prompt', '')
        generator = LLMGenerator(user_prompt, functions)
        res = generator.generate()
        print(res)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()