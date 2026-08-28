# Model Card — Pet Breed Classifier

A 37-way cat and dog breed classifier, built as a measurement exercise
rather than a product.

---

## System description

| | |
|---|---|
| **Task** | Classify a photograph into one of 37 cat or dog breeds |
| **Dataset** | Oxford-IIIT Pet — 7,390 images, official trainval/test split |
| **Shipped model** | **Linear probe** — multinomial logistic regression on frozen ResNet-50 (`IMAGENET1K_V2`) features |
| **Test accuracy** | 0.9283 (macro-F1 0.9273, top-5 0.9962) |
| **Also trained** | Fine-tuned ResNet-50 — 0.9120, *not* shipped (see below) |

### Why the linear probe is the shipped model

The fine-tuned network is the more expensive, more sophisticated option and
it is measurably **worse**: −1.64 accuracy points, p = 1.9 × 10⁻⁴ on a
paired McNemar test over the same 3,669 test images. It is also an order of
magnitude worse calibrated (ECE 0.1023 vs 0.0118). Shipping it because it
sounds more impressive would be choosing the worse model on purpose.

### Intended use

Demonstrating and measuring a transfer-learning pipeline: what pretraining
already provided, whether fine-tuning earned its cost, and whether the
confidence output can be trusted.

### Out of scope

- **Any breed outside the 37.** The model has no "other" class and no
  abstention. Given a Border Collie, a rabbit, or a sofa it returns its most
  confident guess among the 37 — confidently and wrongly.
- **Veterinary, registry, or legal use.** Breed identification carries real
  consequences in some jurisdictions (breed-specific legislation, insurance,
  shelter intake). This model has 8.8% test error, uneven across breeds, and
  is not fit for any of that.
- **Photographs unlike the training distribution.** Oxford-IIIT Pet images
  are well-lit, roughly centred, mostly single-animal. Performance on
  security-camera frames, crowded scenes, or partial views is unmeasured.

---

## How it was evaluated

One pass over the official 3,669-image test split, after all model selection
(linear-probe regularisation strength, fine-tune checkpoint epoch) was done
on a stratified validation split carved out of trainval.

Five models, each ruling out a specific cheap explanation for the one above:
majority class, classical CV features, zero-shot ImageNet, linear probe,
fine-tune. Differences tested with paired McNemar and a paired bootstrap.

### The measurement this project is built around

ImageNet-1k contains named classes for **24 of these 37 breeds** — 21 of 25
dog breeds, 3 of 12 cat breeds. Every metric is therefore reported split on
that boundary, because averaging across it hides the main effect:

| Model | In ImageNet | Not in ImageNet |
|---|---|---|
| ImageNet zero-shot | 0.9849 | 0.0000 |
| Linear probe | 0.9610 | 0.8678 |
| Fine-tuned | 0.9425 | 0.8554 |

The overlap table (`src/data/imagenet_overlap.py`) is hand-written and
unit-tested against torchvision's ImageNet category list, because automatic
name matching gets it wrong in both directions.

---

## What it does well

- **0.9283 test accuracy, 0.9962 top-5**, from minutes of CPU on cached
  features. No GPU training, no backpropagation through the network.
- **Well calibrated.** ECE 0.0118 — it says 93.9% and is right 92.8% of the
  time. Its confidence output is usable as a confidence output, which is not
  true of the fine-tuned alternative.
- **Cheap to reproduce.** Feature extraction is a single forward pass over
  7,350 images, cached to disk; retraining the classifier is seconds.

---

## What it does badly, and known limitations

### 1. It is much worse on cats than on dogs

0.8639 on cats against 0.9590 on dogs. This is not because cats are
intrinsically harder — it is because ImageNet-1k, which supplies every
feature this model uses, names 21 of 25 dog breeds here and 3 of 12 cat
breeds. Anyone deploying this on a cat-heavy population should expect the
cat number, not the headline.

### 2. Accuracy on unseen-by-ImageNet breeds is ~9 points lower

0.9610 on breeds ImageNet named, 0.8678 on breeds it did not. The headline
0.9283 is a weighted average of the two, and the weighting is a property of
this dataset (64.9% / 35.1%), not of any deployment.

### 3. No abstention

The model always answers. There is no "not a pet", no "breed not in the
list", no confidence threshold enforced in the pipeline. The Streamlit app
warns below 50% confidence, but that is a UI courtesy, not a guarantee — and
the reliability diagram shows the low-confidence bands are sparsely
populated, so that threshold is not well estimated.

### 4. The fine-tune result is one training run, not a law

"Fine-tuning made it worse" is measured here, on this dataset, with this
recipe (AdamW, head lr 1e-3, backbone lr 1e-4, label smoothing 0.1, 10
epochs, batch 32, one seed). A different schedule — layer-wise learning-rate
decay, fewer unfrozen blocks, more aggressive augmentation, early stopping
on a larger validation set — could plausibly beat the probe. What the result
does establish is that fine-tuning is **not automatically better**, and that
a project which had only trained the fine-tune would have shipped a worse
model without ever knowing.

### 5. Validation was too small to select on reliably

551 validation images across 37 classes is ~15 per class. Picking the best
of ten epochs on that set selects partly for noise, and it did: the
fine-tune's validation advantage (0.9528 vs 0.9383) reversed on test
(0.9120 vs 0.9283). A larger validation split, or cross-validation, would
have been the right call and is the clearest methodological improvement
available here.

### 6. The fine-tune's miscalibration is partly self-inflicted

Label smoothing at 0.1 caps how confident the model can become, and nothing
downstream recalibrates it. That is why its mean confidence (0.81) sits
below its accuracy (0.91). Temperature scaling on the validation split
would likely fix most of the 0.1023 ECE — it was not run.

### 7. One contested entry in the overlap table

American Pit Bull Terrier is counted as **absent** from ImageNet-1k, though
ImageNet has the closely related American Staffordshire Terrier. Registries
treat them as distinct breeds; counting it as present would move one class
(roughly 100 test images) across the boundary the headline finding rests on.
The judgment is documented in `src/data/imagenet_overlap.py` and the finding
is not sensitive to it at this magnitude, but it is a judgment.

### 8. Conclusions are specific to ImageNet-1k pretraining

A backbone pretrained on a different corpus — LAION, JFT, a
self-supervised objective — would have a different overlap with these
breeds and different numbers. Nothing here generalises to "pretraining
overlap doesn't matter" or "it always matters this much"; the point is that
it is measurable and usually goes unmeasured.

---

## Reproducing

```bash
pip install -r requirements.txt
python -m src.evaluation.run_baselines                # ~3 min, no training
python -m src.models.run_training --stage probe       # ~2 min
python -m src.models.run_training --stage finetune    # 3.3 min on MPS
python -m src.evaluation.run_eval                     # the single test pass
pytest tests/ -q
```

The dataset downloads itself; features and logits are cached under
`data/cache/`. Full results in `output/reports/`.

---

## Author

**Manuel Corona**
