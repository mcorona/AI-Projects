.PHONY: help install install-dev clean data-download train evaluate test lint format demo api

help:
	@echo "Financial Sentiment Analysis - Available Commands"
	@echo "=================================================="
	@echo "install              Install dependencies"
	@echo "install-dev          Install dev + test dependencies"
	@echo "data-download        Download datasets"
	@echo "train                Train the model"
	@echo "evaluate             Evaluate on test set"
	@echo "test                 Run tests"
	@echo "lint                 Lint code (flake8)"
	@echo "format               Format code (black)"
	@echo "demo                 Run Streamlit demo"
	@echo "api                  Run FastAPI server"
	@echo "clean                Remove artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

data-download:
	python -m src.data.loader --download-all

train:
	python -m src.models.train --config configs/training_config.yaml

evaluate:
	python -m src.models.evaluate --model-path output/models/best.pt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=120

format:
	black src/ tests/ notebooks/

demo:
	streamlit run app.py --server.port 8501

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ dist/ build/ *.egg-info/
	rm -rf output/models/* output/reports/* output/figures/*

notebook-eda:
	jupyter notebook notebooks/01_eda_financial_data.ipynb

notebook-bias:
	jupyter notebook notebooks/02_dataset_analysis_bias.ipynb

notebook-training:
	jupyter notebook notebooks/03_model_training.ipynb

notebook-eval:
	jupyter notebook notebooks/04_evaluation_ablation.ipynb
