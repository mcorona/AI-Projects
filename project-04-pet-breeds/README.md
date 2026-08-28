# Pet Breeds, and What ImageNet Already Knew

## 📊 Project Overview

Fine-grained image classification of 37 cat and dog breeds — built to
measure something most transfer-learning writeups leave unexamined:
**how much of the task the pretrained model had already been trained on.**

The standard result on this benchmark is "transfer learning gets you 90%+".
That is true and it is misleading. ImageNet-1k spends roughly 120 of its
1,000 classes on dog breeds and 5 on domestic cats, and 24 of these 37
breeds are literally ImageNet classes. A pretrained classifier hasn't
*generalised* to those breeds — it was trained to name them.

This project makes that measurable by splitting every metric on one
boundary: breeds ImageNet already names, against breeds it has never been
given a name for.

**Dataset:** [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
— 7,390 images, 37 breeds (12 cat, 25 dog), with an official trainval/test
split. Downloads itself via torchvision; no Kaggle account.

---

## 📈 Results

Official test split, 3,669 images, 37 breeds. One pass, after all model
selection was done on a held-out validation split.

| Model | Accuracy | Macro F1 | Top-5 | **In ImageNet** | **Not in ImageNet** | Cats | Dogs |
|---|---|---|---|---|---|---|---|
| Majority class (floor) | 0.0267 | 0.0014 | — | 0.0000 | 0.0762 | 0.0828 | 0.0000 |
| HOG + colour histogram → linear SVM | 0.0905 | 0.0897 | — | 0.0961 | 0.0801 | 0.0947 | 0.0885 |
| ImageNet classifier, **zero-shot** | 0.6397 | 0.5561 | — | **0.9849** | **0.0000** | 0.2502 | 0.8250 |
| **Linear probe on frozen features** | **0.9283** | **0.9273** | 0.9962 | 0.9610 | 0.8678 | 0.8639 | 0.9590 |
| Fine-tuned ResNet-50 | 0.9120 | 0.9104 | 0.9935 | 0.9425 | 0.8554 | 0.8639 | 0.9348 |

ImageNet-1k names 24 of these 37 breeds — **21 of 25 dogs, but only 3 of 12
cats.** Those 24 breeds are 64.9% of the test split.

---

## 🔍 The finding

### 1. Two thirds of this benchmark was already an ImageNet task

A stock ImageNet classifier, with **zero gradient steps on this dataset**,
scores **98.49%** on the 24 breeds ImageNet already names — and 0.00% on the
other 13, which it structurally cannot express (it has no output unit for
them). Its 63.97% overall is almost entirely the first number diluted by
the second.

So "transfer learning reaches 90%+ on Oxford Pets" is true and misleading.
Most of the apparent difficulty of this benchmark was paid for in advance
by someone else's labelling effort, and a single headline accuracy averages
across that boundary as if it weren't there.

The comparison that makes it concrete: hand-designed features (HOG + colour
histogram, a fair representative of pre-deep-learning computer vision) get
**9.05%**. The task is genuinely hard. The pretrained model just happened to
have been trained on most of it.

### 2. Fine-tuning made the model worse — and exactly where you'd predict

| Comparison | Δ accuracy | 95% CI | p (McNemar) |
|---|---|---|---|
| Fine-tune vs linear probe, **overall** | **−0.0164** | [−0.0251, −0.0082] | 1.9 × 10⁻⁴ |
| … on breeds **ImageNet already knew** | **−0.0185** | [−0.0264, −0.0105] | 4.9 × 10⁻⁶ |
| … on breeds **it never knew** | −0.0124 | [−0.0319, +0.0062] | 0.24 (n.s.) |

Two orders of magnitude more compute bought a **significantly worse** model.
And the damage is concentrated in the breeds the pretrained network already
knew, while on the breeds it never knew the difference is not significant.

That is catastrophic forgetting in miniature: updating every weight on
3,129 images degraded pretrained knowledge that was already better than
anything those 3,129 images could teach, and bought nothing measurable in
return.

**Validation would not have caught this.** The fine-tune won on validation
(95.28% vs the probe's 93.83%) and lost on test:

| | Validation | Test | Gap |
|---|---|---|---|
| Linear probe | 0.9383 | 0.9283 | −0.010 |
| Fine-tuned ResNet-50 | 0.9528 | 0.9120 | **−0.041** |

Model selection ran on 551 validation images. Picking the best of ten
epochs on a set that small selects partly for noise, and the fine-tune's
apparent advantage did not survive contact with 3,669 held-out images.
Training accuracy hit 99.9% by epoch 5 — the model had memorised the
training set and validation was too small to say so.

### 3. The cheap model is also the better-calibrated one

| | ECE | Mean confidence | Accuracy |
|---|---|---|---|
| Linear probe | **0.0118** | 0.9391 | 0.9283 |
| Fine-tuned ResNet-50 | 0.1023 | 0.8097 | 0.9120 |

The linear probe is nearly perfectly calibrated — it says 93.9% and is right
92.8% of the time. The fine-tune is off by nearly ten points, in the
*under*confident direction, which is a direct consequence of the label
smoothing (0.1) used during training: smoothing caps how confident the model
can get, and nothing afterwards recalibrates it. That was my choice, and it
shows up as a real cost.

### 4. The cat deficit tracks the ImageNet deficit

Both transfer models score ~0.864 on cats and 0.935–0.959 on dogs. That gap
is not because cats are intrinsically harder to tell apart — it is because
ImageNet-1k names 21 of 25 dog breeds here and 3 of 12 cat breeds. The model
is better at dogs because it was pretrained to be.

---

## What this project actually argues

The headline number on this benchmark is uninformative on its own. Three
things have to be reported alongside it for it to mean anything:

1. **What the pretrained model already knew** — measured by a zero-shot
   baseline, not assumed.
2. **Whether the expensive option beat the cheap one** — with a paired
   significance test, because here it lost by an amount a decimal comparison
   would have called a rounding error.
3. **Whether the confidence is usable** — accuracy says nothing about it, and
   the two models here differ by an order of magnitude in ECE.

---

## 🛠️ Tech Stack

| Component | Tool | Reason |
|---|---|---|
| **Backbone** | torchvision ResNet-50 (`IMAGENET1K_V2`) | The V2 recipe (80.9% ImageNet top-1) — using V1 would understate the off-the-shelf baseline and flatter the fine-tune |
| **Classical baseline** | scikit-image HOG + colour histogram → LinearSVC | What image classification looked like before deep learning |
| **Zero-shot baseline** | Raw 1000-way ImageNet head, mapped to breeds | Measures how much of the task was already solved |
| **Linear probe** | scikit-learn logistic regression on frozen 2048-d features | The cheap option a team reaches for first |
| **Fine-tune** | PyTorch, AdamW, two learning rates, cosine schedule | The expensive option, which has to justify itself |
| **Calibration** | Expected calibration error | Accuracy doesn't say whether the model knows when it's wrong |
| **Dashboard** | Streamlit | Upload a photo; see the breed, the confidence, and whether ImageNet knew it |

Why two learning rates: the classification head is random and needs to move
fast, the backbone is already good and mostly needs nudging. A single
learning rate high enough to train the head from scratch wrecks the
pretrained features in the first few hundred steps — the classic way a
fine-tune ends up *below* a linear probe.

---

## 📁 Repository Layout

```
project-04-pet-breeds/
├── src/
│   ├── data/
│   │   ├── loader.py               Official split, stratified val, transforms
│   │   └── imagenet_overlap.py     The hand-verified Pet → ImageNet-1k table
│   ├── models/
│   │   ├── backbone.py             Pretrained features and raw ImageNet logits
│   │   ├── baselines.py            Majority, classical CV, zero-shot
│   │   ├── train.py                Linear probe + fine-tune
│   │   └── run_training.py         Training driver (never reads test)
│   └── evaluation/
│       ├── metrics.py              Accuracy, macro-F1, group splits, ECE
│       ├── run_baselines.py        Baseline driver
│       └── run_eval.py             The single pass over the test split
├── notebooks/
│   ├── 01_eda.ipynb                The dataset and the overlap
│   └── 02_results.ipynb            The ladder, the split, and the failures
├── tests/                          Dataset contract + overlap-table guards
├── app.py                          Streamlit dashboard
├── Dockerfile                      Serves the dashboard
└── MODEL_CARD.md                   What this classifier cannot do
```

---

## 🚀 Setup

```bash
cd project-04-pet-breeds
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The dataset (792 MB) downloads itself on first use:

```bash
python -m src.data.loader
python -m src.data.imagenet_overlap    # prints the 24/37 coverage
```

### Reproducing the results

```bash
python -m src.evaluation.run_baselines               # ~3 min, no training
python -m src.models.run_training --stage probe      # ~2 min
python -m src.models.run_training --stage finetune   # 10 epochs, 3.3 min on an M-series Mac (MPS)
python -m src.evaluation.run_eval                    # the one pass over test
```

Backbone features, ImageNet logits, and classical features are all cached
under `data/cache/`, so re-running scores in seconds.

### Dashboard

```bash
streamlit run app.py
```

Or via Docker (the checkpoint is mounted, not baked in):

```bash
docker build -t pet-breeds .
docker run -p 8501:8501 -v "$(pwd)/output/models:/app/output/models" pet-breeds
```

### Tests

```bash
pytest tests/ -q
```

The overlap table is the load-bearing artifact of this project, so it is
checked against torchvision's actual ImageNet category list rather than
trusted — including spot checks on the breeds ImageNet lists under older
names (`Japanese Chin` → `Japanese spaniel`, `Leonberger` → `Leonberg`,
`Scottish Terrier` → `Scotch terrier`).

---

## ⚠️ Limitations

Full writeup in [`MODEL_CARD.md`](MODEL_CARD.md). The short version:

- **37 breeds and nothing else.** Given a breed outside the list, a rabbit,
  or a sofa, the model still returns its most confident guess among the 37.
- **One judgment call in the overlap table.** American Pit Bull Terrier is
  counted as *absent* from ImageNet, though ImageNet has the closely related
  American Staffordshire Terrier. Documented in `imagenet_overlap.py`.
- **Conclusions are about ImageNet-1k pretraining specifically.** A backbone
  pretrained on a different corpus would have a different overlap and
  different numbers.

---

## Author

**Manuel Corona**
