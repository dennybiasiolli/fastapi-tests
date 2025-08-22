init:
	uv sync --dev
	pre-commit install

style-fix:
	pre-commit run --all-files

style-check:
	pre-commit run --all-files --check
