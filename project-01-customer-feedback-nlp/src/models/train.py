"""
Training entrypoint for FinancialSentimentClassifier.

Usage:
    python -m src.models.train --config configs/training_config.yaml

Author: Manuel Corona
"""

import argparse
import math

import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.preprocessing import TokenizedFinancialDataset, clean_text
from src.models.sentiment_classifier import FinancialSentimentClassifier, ModelConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict, tokenizer, max_length: int):
    train_df = pd.read_parquet(cfg["data"]["train_path"])
    val_df = pd.read_parquet(cfg["data"]["val_path"])

    train_ds = TokenizedFinancialDataset(
        train_df["text"].apply(clean_text).tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    val_ds = TokenizedFinancialDataset(
        val_df["text"].apply(clean_text).tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["test_batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    return train_loader, val_loader, len(train_ds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument(
        "--model-name-or-path", default=None,
        help="Override config's model.model_name (e.g. a local directory for offline runs).",
    )
    parser.add_argument("--fast-dev-run", action="store_true", help="Run 1 batch through the pipeline as a smoke test.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override config's training.epochs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg["seed"])

    model_name = args.model_name_or_path or cfg["model"]["model_name"]
    max_length = cfg["model"]["max_length"]
    epochs = args.max_epochs or cfg["training"]["epochs"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, val_loader, n_train = build_dataloaders(cfg, tokenizer, max_length)

    # The config's num_training_steps/warmup_steps are placeholders sized for a much
    # larger corpus; compute the real schedule from this dataset instead.
    steps_per_epoch = math.ceil(n_train / cfg["data"]["batch_size"])
    num_training_steps = steps_per_epoch * epochs
    warmup_steps = min(cfg["training"]["warmup_steps"], max(1, num_training_steps // 10))

    model_config = ModelConfig(
        model_name=model_name,
        num_classes=cfg["model"]["num_classes"],
        max_length=max_length,
        dropout=cfg["model"]["dropout"],
        # PyYAML's default resolver doesn't reliably parse exponential notation
        # (e.g. "2e-5") as a float, so cast explicitly.
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    model = FinancialSentimentClassifier(model_config)

    checkpoint_cb = ModelCheckpoint(
        dirpath="output/models",
        filename="best-{epoch}-{val_f1_weighted:.3f}",
        monitor="val_f1_weighted",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor=cfg["validation"]["early_stopping_metric"],
        mode=cfg["validation"]["early_stopping_mode"],
        patience=cfg["validation"]["early_stopping_patience"],
    )
    logger = CSVLogger("output/reports", name="training_logs")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        gradient_clip_val=cfg["training"]["max_grad_norm"],
        accumulate_grad_batches=cfg["training"]["gradient_accumulation_steps"],
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if not args.fast_dev_run:
        print(f"Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
