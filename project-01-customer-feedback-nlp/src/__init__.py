"""Financial Sentiment Analysis Package"""

__version__ = "1.0.0"
__author__ = "Manuel Corona"

from src.models.sentiment_classifier import FinancialSentimentClassifier, SentimentInference

__all__ = [
    "FinancialSentimentClassifier",
    "SentimentInference",
]
