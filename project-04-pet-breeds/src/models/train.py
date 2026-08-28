"""
The two transfer-learning models: a linear probe and a full fine-tune.

They answer different questions and both are worth running:

  linear_probe   Freeze the ImageNet backbone, train only a 37-way linear
                 layer on its 2048-d features. Minutes of CPU, no
                 backpropagation through the network. This is the cheap
                 option a team would actually reach for first, and it sets
                 the bar the expensive option has to clear.

  finetune       Update every weight. Roughly two orders of magnitude more
                 compute. Whether that buys anything is the question, not
                 the assumption.

Model selection is on the held-out validation split carved from trainval.
The test split is read exactly once, by src/evaluation/run_eval.py.

Author: Manuel Corona
"""

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "output" / "models"
REPORTS_DIR = ROOT / "output" / "reports"


# --- Linear probe --------------------------------------------------------

def train_linear_probe(xtr: np.ndarray, ytr: np.ndarray,
                       xval: np.ndarray, yval: np.ndarray, seed: int = 0):
    """
    Multinomial logistic regression on frozen backbone features.

    C is selected on the validation split rather than fixed. With 2048
    features and ~3,100 training images the model is in the regime where
    regularisation strength genuinely matters, and picking C by eye would
    be tuning on nothing.
    """
    from sklearn.linear_model import LogisticRegression

    best = None
    trace = []
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        clf.fit(xtr, ytr)
        acc = float((clf.predict(xval) == yval).mean())
        trace.append({"C": C, "val_accuracy": round(acc, 4)})
        print(f"  linear probe C={C:<7} val acc={acc:.4f}")
        if best is None or acc > best[0]:
            best = (acc, C, clf)
    val_acc, C, clf = best
    print(f"  selected C={C} (val acc={val_acc:.4f})")
    return clf, {"selected_C": C, "val_accuracy": round(val_acc, 4), "search": trace}


# --- Full fine-tune ------------------------------------------------------

def _run_epoch(model, loader, device, criterion, optimizer=None, desc=""):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def finetune(num_classes: int, train_loader, val_loader, device: str,
             epochs: int = 10, head_lr: float = 1e-3, backbone_lr: float = 1e-4,
             label_smoothing: float = 0.1, seed: int = 0) -> Dict:
    """
    Fine-tune the whole ResNet-50.

    Two learning rates, not one: the head is random and needs to move fast,
    the backbone is already good and mostly needs nudging. A single lr high
    enough to train the head from scratch would wreck the pretrained
    features in the first few hundred steps -- which is the classic way a
    fine-tune ends up *below* a linear probe.

    Label smoothing at 0.1 because several of these breeds are genuinely
    near-identical in photographs, and a loss that demands full confidence
    on an ambiguous pair mostly teaches overconfidence.
    """
    from src.models.backbone import build_backbone

    torch.manual_seed(seed)
    model, _ = build_backbone(num_classes=num_classes)
    model.to(device)

    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": backbone_lr},
         {"params": head, "lr": head_lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS_DIR / "resnet50_finetuned.pt"
    history, best_val, best_epoch = [], -1.0, -1
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, device, criterion,
                                     optimizer, f"epoch {epoch} train")
        va_loss, va_acc = _run_epoch(model, val_loader, device, criterion,
                                     None, f"epoch {epoch} val")
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "train_accuracy": round(tr_acc, 4),
                        "val_loss": round(va_loss, 4),
                        "val_accuracy": round(va_acc, 4)})
        flag = ""
        if va_acc > best_val:
            # Checkpoint on validation accuracy, not on the last epoch. The
            # last epoch is not reliably the best one and selecting it would
            # be a silent coin flip.
            best_val, best_epoch = va_acc, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_accuracy": va_acc, "num_classes": num_classes}, ckpt)
            flag = "  <- best, checkpointed"
        print(f"epoch {epoch:2d}  train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"val {va_loss:.4f}/{va_acc:.4f}{flag}")

    summary = {"best_val_accuracy": round(best_val, 4), "best_epoch": best_epoch,
               "epochs": epochs, "head_lr": head_lr, "backbone_lr": backbone_lr,
               "label_smoothing": label_smoothing, "seed": seed,
               "minutes": round((time.time() - t0) / 60, 1), "history": history}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "finetune_history.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbest val accuracy {best_val:.4f} at epoch {best_epoch} "
          f"({summary['minutes']} min)")
    return summary


@torch.no_grad()
def predict_proba(model, loader, device: str) -> np.ndarray:
    model.eval().to(device)
    out = []
    for x, _ in tqdm(loader, desc="predicting", leave=False):
        out.append(torch.softmax(model(x.to(device)), dim=1).float().cpu().numpy())
    return np.concatenate(out)


def load_finetuned(device: str):
    from src.models.backbone import build_backbone
    ckpt = torch.load(MODELS_DIR / "resnet50_finetuned.pt", map_location=device)
    model, _ = build_backbone(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
