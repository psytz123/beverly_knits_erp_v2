# Beverly Knits ERP - Makefile

.PHONY: help install run test clean lint format docker-build docker-run sync-data validate setup

# Default target
help:
	@echo "Beverly Knits ERP - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Complete initial setup"
	@echo "  make install      - Install dependencies"
	@echo "  make venv         - Create virtual environment"
	@echo ""
	@echo "Running:"
	@echo "  make run          - Run the ERP server"
	@echo "  make run-dev      - Run in development mode"
	@echo "  make run-docker   - Run with Docker"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run code linters"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Data Management:"
	@echo "  make sync-data    - Sync data from SharePoint"
	@echo "  make validate     - Validate data files"
	@echo "  make backup-data  - Backup production data"
	@echo ""
	@echo "Deployment:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-push  - Push to registry"
	@echo "  make deploy       - Deploy to production"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Clean temporary files"
	@echo "  make clean-all    - Clean everything (including cache)"

# Setup commands
setup: venv install
	@echo "Setup complete! Activate venv with: source venv/bin/activate"

venv:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Running commands
run:
	python src/core/beverly_comprehensive_erp.py

run-dev:
	FLASK_ENV=development FLASK_DEBUG=1 python src/core/beverly_comprehensive_erp.py

run-docker:
	docker-compose up

# Testing commands
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/
	pylint src/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Data management
sync-data:
	python src/data_sync/daily_data_sync.py

validate:
	python scripts/validate_data.py

backup-data:
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf backups/data_backup_$$timestamp.tar.gz data/production/
	@echo "Data backed up to backups/data_backup_$$(date +%Y%m%d_%H%M%S).tar.gz"

# Docker commands
docker-build:
	docker build -t beverly-knits-erp:latest .

docker-push:
	docker tag beverly-knits-erp:latest registry.example.com/beverly-knits-erp:latest
	docker push registry.example.com/beverly-knits-erp:latest

docker-run:
	docker run -p 5005:5005 --env-file config/.env beverly-knits-erp:latest

# Deployment
deploy:
	@echo "Deploying to production..."
	./deployment/scripts/deploy.sh

deploy-staging:
	@echo "Deploying to staging..."
	./deployment/scripts/deploy.sh staging

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf build dist

clean-all: clean
	rm -rf data/cache/*
	rm -rf logs/*
	rm -rf venv/

# Database
db-migrate:
	python scripts/migrate_database.py

db-backup:
	python scripts/backup_database.py

# Monitoring
monitor:
	@echo "Starting monitoring dashboard..."
	python -m webbrowser http://localhost:5005/consolidated

logs:
	tail -f logs/app.log

# Development helpers
shell:
	python -i -c "from src.core.beverly_comprehensive_erp import *"

debug:
	python -m pdb src/core/beverly_comprehensive_erp.py

# Performance
profile:
	python -m cProfile -o profile.stats src/core/beverly_comprehensive_erp.py
	python -m pstats profile.stats

benchmark:
	python tests/performance/benchmark.py