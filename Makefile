.PHONY: install install-dev test test-cov ingest query companies check-manifest reset scrape-docs clean

VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

test:
	PYTHONPATH=src $(VENV)/bin/pytest tests/ -v

test-cov:
	PYTHONPATH=src $(VENV)/bin/pytest tests/ -v --cov=src/rag_platform --cov-report=term-missing

# Usage: make ingest COMPANY=company_a
ingest:
	PYTHONPATH=src $(PYTHON) scripts/ingest_company_pdfs.py --company $(COMPANY)

# Usage: make query COMPANY=company_a QUERY="What is the deployment process?"
query:
	PYTHONPATH=src $(PYTHON) scripts/query_company.py --company $(COMPANY) --query "$(QUERY)"

companies:
	PYTHONPATH=src $(PYTHON) scripts/manage_companies.py list

# Usage: make check-manifest COMPANY=company_a
check-manifest:
	PYTHONPATH=src $(PYTHON) scripts/check_manifest.py --company $(COMPANY)

# Usage: make reset COMPANY=company_a
reset:
	PYTHONPATH=src $(PYTHON) scripts/reset_data.py --company $(COMPANY)

# Usage: make scrape-docs URL=https://example.com/docs PREFIX=/docs OUT=data/companies/example/raw_pdfs
scrape-docs:
	$(PYTHON) scripts/scrape_docs_to_pdf.py --url $(URL) --prefix $(PREFIX) --out $(OUT)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage