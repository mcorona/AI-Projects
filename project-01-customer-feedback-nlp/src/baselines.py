"""
Baseline sentiment classifiers for comparison against the fine-tuned model.

Three baselines, increasing in sophistication:
1. TextBlob (rule-based lexicon)
2. TF-IDF + Linear SVM (traditional ML)
3. DistilBERT fine-tuned on SST-2 (pre-trained, generic domain, no fine-tuning
   on financial text) -- demonstrates the domain-shift gap this project aims
   to close.

Author: Manuel Corona
"""

from typing import List, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# --- Baseline 1: TextBlob (rule-based) ---------------------------------

def textblob_sentiment(text: str) -> str:
    """Classify sentiment polarity with TextBlob's built-in lexicon."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"


def textblob_predict(texts: List[str]) -> List[str]:
    return [textblob_sentiment(t) for t in texts]


# --- Baseline 2: TF-IDF + Linear SVM (traditional ML) -------------------

def build_tfidf_svm_pipeline(max_features: int = 5000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features)),
        ("svm", LinearSVC(max_iter=1000)),
    ])


def tfidf_svm_fit_predict(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    max_features: int = 5000,
) -> List[str]:
    pipeline = build_tfidf_svm_pipeline(max_features=max_features)
    pipeline.fit(train_texts, train_labels)
    return list(pipeline.predict(test_texts))


# --- Baseline 3: DistilBERT (pre-trained on SST-2, generic domain) ------

class DistilBertSST2Baseline:
    """
    Wraps a DistilBERT model fine-tuned on SST-2 (binary NEGATIVE/POSITIVE,
    generic movie-review domain -- NOT financial text, and NOT 3-class).

    `model_path` defaults to the standard HuggingFace Hub id and works with
    `AutoTokenizer`/`AutoModelForSequenceClassification` out of the box on
    any machine with normal internet access. Pass a local directory instead
    if running somewhere the Hub is unreachable.
    """

    def __init__(self, model_path: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, texts: List[str], batch_size: int = 32, mapped_to_neutral: Optional[float] = 0.6) -> List[str]:
        """
        Predict sentiment. Since SST-2 is binary, there's no native
        "neutral" class -- as a simple proxy, predictions with low
        confidence (max softmax probability below `mapped_to_neutral`)
        are relabeled "neutral" so predictions are comparable to the
        3-class financial labels. Set to None to disable and always
        return positive/negative.
        """
        predictions = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=1)
                confs, preds = probs.max(dim=1)
                for conf, pred in zip(confs.tolist(), preds.tolist()):
                    label = self.model.config.id2label[pred].lower()
                    if mapped_to_neutral is not None and conf < mapped_to_neutral:
                        label = "neutral"
                    predictions.append(label)
        return predictions
