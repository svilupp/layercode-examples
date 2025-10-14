.PHONY: help server tunnel format lint lint-fix typecheck test check check-all install dev clean

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  make server      - Start the FastAPI server with auto-reload"
	@echo "  make tunnel      - Start the Layercode tunnel"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format      - Format code with ruff"
	@echo "  make lint        - Lint code with ruff (check only)"
	@echo "  make lint-fix    - Lint and auto-fix issues with ruff"
	@echo "  make typecheck   - Run mypy type checking"
	@echo "  make test        - Run pytest tests"
	@echo "  make check       - Run format + lint + typecheck"
	@echo "  make check-all   - Run format + lint + typecheck + tests"
	@echo ""
	@echo "Setup:"
	@echo "  make install     - Install production dependencies"
	@echo "  make dev         - Install all dependencies including dev tools"
	@echo "  make clean       - Remove Python cache files"

# Start the FastAPI server with uvicorn
server:
	uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start the Layercode tunnel (reads agent_id from layercode.config.json)
tunnel:
	npx @layercode/cli tunnel --port=8000

# Format code with ruff
format:
	@echo "Formatting code with ruff..."
	uv run ruff format .

# Lint code with ruff (check only)
lint:
	@echo "Linting code with ruff..."
	uv run ruff check .

# Lint and auto-fix issues with ruff
lint-fix:
	@echo "Linting and fixing code with ruff..."
	uv run ruff check --fix .

# Type check with mypy
typecheck:
	@echo "Type checking with mypy..."
	uv run mypy main.py layercode_sdk.py

# Run tests with pytest
test:
	@echo "Running tests with pytest..."
	uv run pytest

# Run format, lint, and typecheck
check: format lint typecheck
	@echo ""
	@echo "✓ All checks passed!"

# Run all checks including tests
check-all: format lint typecheck test
	@echo ""
	@echo "✓ All checks and tests passed!"

# Install production dependencies only
install:
	uv sync

# Install all dependencies including dev tools
dev:
	uv sync --extra dev

# Clean Python cache files
clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete!"
