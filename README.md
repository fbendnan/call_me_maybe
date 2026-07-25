*This project has been created as part of the 42 curriculum by fbendnan*

## Description
A function‑calling system that translates natural language prompts into structured JSON using **constrained decoding**.  
It guarantees 100% valid JSON output even with a small 0.6B parameter model.

## Instructions
- **Install**: `make install` or `uv sync`
- **Run**: `make run` or `uv run python -m src [--options]`
- **Debug**: `make debug`
- **Clean**: `make clean`

## Algorithm Explanation
We define a JSON **skeleton** and force each token via **logit masking**:
- Literal parts are forced exactly.
- Placeholders (prompt, function name, parameters) are generated under token‑level restrictions.
- Generation stops immediately after the final `}`.

## Design Decisions
- **Token caching** avoids repeated encoding of literals.
- **Vocabulary lookup** replaces `decode()` for faster token‑to‑string conversion.
- **Pydantic** validates the final output against the expected schema.
- **Argparse** allows custom input/output paths.

## Performance Analysis
All 11 test prompts are processed in **under 4 minutes** (≈2 seconds per prompt) on standard hardware.

## Challenges Faced
- Handling multi‑token literals (e.g., `"fn_add_numbers"`).
- Ensuring the loop stops after the JSON is complete.
- Balancing speed with correctness.

## Testing Strategy
- Validated against the provided `function_calling_tests.json`.
- Output is checked with `json.loads()` and Pydantic models.
- Edge cases (empty prompts, malformed inputs) are handled gracefully.

## Resources
