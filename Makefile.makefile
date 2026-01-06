# Makefile for Semantic Search Engine

.PHONY: help install dev test clean build run deploy

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
VENV := venv

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  dev         Set up development environment"
	@echo "  test        Run tests"
	@echo "  clean       Clean up temporary files"
	@echo "  build       Build Docker image"
	@echo "  run         Run application locally"
	@echo "  deploy      Deploy to production"
	@echo "  docs        Generate documentation"
	@echo "  benchmark   Run performance benchmarks"
	@echo "  evaluate    Run evaluation"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install faiss-cpu

dev:
	$(PIP) install -r requirements-dev.txt
	cp .env.example .env
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install pre-commit
	pre-commit install

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf data/indices/*
	rm -rf data/embeddings/*
	rm -rf logs/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

build:
	$(DOCKER_COMPOSE) build

run:
	$(PYTHON) app.py

run-docker:
	$(DOCKER_COMPOSE) up -d
	@echo "Application running at http://localhost:5000"
	@echo "Elasticsearch at http://localhost:9200"

stop-docker:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f

deploy:
	@echo "Building Docker image..."
	$(DOCKER) build -t semantic-search-engine:latest .
	@echo "Pushing to registry..."
	# Add your registry push commands here

index-example:
	$(PYTHON) index_codebase.py --repo-path ./example_codebase --force

benchmark:
	$(PYTHON) scripts/benchmark.py

evaluate:
	$(PYTHON) scripts/evaluate.py

docs:
	$(PYTHON) -m pydoc -w src
	@echo "Documentation generated"

format:
	$(PYTHON) -m black src/ api/ tests/ scripts/
	$(PYTHON) -m isort src/ api/ tests/ scripts/

lint:
	$(PYTHON) -m flake8 src/ api/ tests/ scripts/
	$(PYTHON) -m mypy src/ api/ tests/ scripts/

check:
	make format
	make lint
	make test

docker-clean:
	$(DOCKER) system prune -f
	$(DOCKER) volume prune -f

update-deps:
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) freeze > requirements.txt

backup:
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		data/indices/ \
		data/embeddings/ \
		data/chroma_db/ \
		logs/ \
		.env

.PHONY: help
