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
        raw = generator.generate()
        print(raw)

        json_str = extract_json(raw)
        if json_str is None:
            results.append({"prompt": user_prompt, "error": "No valid JSON found"})
            continue

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            results.append({"prompt": user_prompt, "error": "Invalid JSON", "raw": raw})
            continue

        ok, msg = validate_output(data)
        if not ok:
            results.append({"prompt": user_prompt, "error": msg, "raw": data})
        else:
            results.append(data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()