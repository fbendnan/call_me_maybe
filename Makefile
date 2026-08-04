install:
	uv sync

run:
	uv run python3 -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf src/__pycache__ .mypy_cache .pytest_cache data/output/*

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict