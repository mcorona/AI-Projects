"""
Sentiment Classifier Module

This module contains the core sentiment classification model using fine-tuned
transformers. It wraps HuggingFace's transformers library with additional
features for training, evaluation, and inference.

Author: Manuel Corona
Date: 2026-08-22
"""

from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for sentiment classifier model."""
    model_name: str = "roberta-base"
    num_classes: int = 3
    max_length: int = 128
    dropout: float = 0.1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    num_training_steps: int = 15000


class FinancialSentimentClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for financial sentiment classification.

    This module fine-tunes a pre-trained transformer model (RoBERTa) on
    financial sentiment data. It includes:
    - Efficient fine-tuning strategy
    - Multi-class sentiment classification
    - Comprehensive metrics tracking
    - Learning rate scheduling

    Args:
        config (ModelConfig): Model configuration
        num_labels (int): Number of sentiment classes (default: 3)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_classes,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
        )

        # Freeze early layers (layers 0-9 of 12)
        for param in self.model.roberta.encoder.layer[:9].parameters():
            param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics tracking
        self.train_predictions = []
        self.train_labels = []
        self.val_predictions = []
        self.val_labels = []

        # Class labels for reference
        self.class_labels = ["Negative", "Neutral", "Positive"]

        self.save_hyperparameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Logits [batch_size, num_classes]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Track for metrics
        preds = logits.argmax(dim=1)
        self.train_predictions.extend(preds.cpu().numpy())
        self.train_labels.extend(labels.cpu().numpy())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Track for metrics
        preds = logits.argmax(dim=1)
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_labels.extend(labels.cpu().numpy())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Calculate validation metrics at epoch end."""
        if len(self.val_predictions) == 0:
            return

        predictions = np.array(self.val_predictions)
        labels = np.array(self.val_labels)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
        precision = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(labels, predictions, average="weighted", zero_division=0)

        # Log metrics
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1_weighted", f1_weighted, on_epoch=True, prog_bar=True)
        self.log("val_precision", precision, on_epoch=True)
        self.log("val_recall", recall, on_epoch=True)

        # Clear for next epoch
        self.val_predictions.clear()
        self.val_labels.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        self.log("test_loss", loss, on_epoch=True)

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Prediction step with confidence scores.

        Args:
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index

        Returns:
            Dictionary with predictions and probabilities
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        logits = self.forward(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        return {
            "predictions": predictions.cpu().numpy(),
            "probabilities": probabilities.cpu().numpy(),
            "logits": logits.cpu().numpy(),
        }


class SentimentInference:
    """
    Inference wrapper for sentiment classification.

    Handles tokenization, batch processing, and post-processing of model outputs.
    """

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize inference module.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to use (cuda or cpu)
        """
        self.device = device
        self.model = FinancialSentimentClassifier.load_from_checkpoint(model_path)
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.model_name)
        self.class_labels = ["Negative", "Neutral", "Positive"]

    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        return_probabilities: bool = True,
    ) -> Dict[str, Union[List[str], np.ndarray]]:
        """
        Predict sentiment for input texts.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for inference
            return_probabilities: Whether to return confidence scores

        Returns:
            Dictionary with predictions and metadata
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        predictions = []
        confidences = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.model.config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                # Move to device
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                # Get confidence (max probability)
                max_probs = probs.max(dim=1).values

                predictions.extend([self.class_labels[p] for p in preds.cpu().numpy()])
                if return_probabilities:
                    confidences.extend(max_probs.cpu().numpy())

        result = {
            "texts": texts,
            "predictions": predictions,
        }

        if return_probabilities:
            result["confidence"] = confidences

        return result

    def predict_batch_with_details(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Predict with detailed output including all class probabilities.

        Args:
            texts: List of texts to predict
            batch_size: Batch size for inference

        Returns:
            List of dictionaries with detailed predictions
        """
        detailed_results = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.model.config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                for j, text in enumerate(batch_texts):
                    result = {
                        "text": text,
                        "prediction": self.class_labels[preds[j].item()],
                        "confidence": probs[j, preds[j]].item(),
                        "probabilities": {
                            label: prob.item()
                            for label, prob in zip(self.class_labels, probs[j])
                        }
                    }
                    detailed_results.append(result)

        return detailed_results
