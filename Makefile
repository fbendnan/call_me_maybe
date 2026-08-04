install:
	uv sync

run:
	uv run python3 -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf src/__pycache__ .mypy_cache .pytest_cache data/output/*

lint:
	flake8 --exclude=.venv,__pycache__,moulinette,llm_sdk .
	mypy src

lint-strict:
	flake8 --exclude=.venv,__pycache__,moulinette,llm_sdk .
	mypy --follow-imports=skip --strict src
