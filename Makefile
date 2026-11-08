PYTHON ?= python

.PHONY: install lint typecheck test check bench demo-t2 demo-t4 run-t1 clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy task1_mcp_server task2_mcp_gateway task3_llm_guardrail task4_model_router

test:
	$(PYTHON) -m pytest

check: lint typecheck test

bench:
	$(PYTHON) -m task3_llm_guardrail.bench

run-t1:
	$(PYTHON) -m task1_mcp_server.server

demo-t2:
	PYTHON=$(PYTHON) bash scripts/demo_task2.sh

demo-t4:
	PYTHON=$(PYTHON) bash scripts/demo_task4.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis data build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
