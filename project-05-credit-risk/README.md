# Credit Decisions, Priced

## 📊 Project Overview

Default prediction on 30,000 credit-card accounts — built to measure the
thing that sits between a model and a business: **the threshold, and what
the two mistakes actually cost.**

The standard writeup on this dataset reports AUC, or accuracy, and stops. A
credit model does not emit a score for its own sake. Someone approves or
declines, and declining a customer who would have paid costs the margin on
their balance while approving one who defaults costs the balance. Those are
not the same number, and a 0.5 cutoff is the buried claim that they are.

This project makes the decision the unit of measurement. Every model is
scored twice — once by how well it ranks, once by what its decisions cost —
and the two answers disagree.

**Dataset:** [UCI "default of credit card clients"](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
— 30,000 Taiwanese accounts, April–September 2005, 22.12% default in
October. Downloads itself; 5 MB; no account needed.

---

## 📈 Results

Held-out test split, 7,500 accounts, NT$380.5M of balance at risk. One pass,
after all model selection and every threshold was frozen on validation.

| Model | AUC | ECE | Threshold | Cost | Cost at 0.5 | **Saved vs no model** |
|---|---|---|---|---|---|---|
| Base rate (floor) | 0.5000 | 0.0000 | — | 5841.0 | 12442.5 | **0.0** |
| Delinquency rule (one column) | 0.7205 | 0.0119 | — | 5841.0 | 8434.5 | **0.0** |
| Logistic regression | 0.7377 | 0.0535 | 0.063 | 5637.0 | 9597.5 | +204.0 |
| Logistic, `class_weight="balanced"` | 0.7371 | 0.2379 | 0.218 | 5662.5 | 5832.0 | +178.5 |
| Random forest | 0.7926 | 0.0234 | 0.129 | 4695.0 | 8061.0 | +1146.0 |
| **Gradient boosting** | **0.7881** | **0.0115** | **0.102** | **4712.5** | 7933.0 | **+1128.5** |
| Gradient boosting, isotonic | 0.7913 | 0.0126 | 0.114 | 4736.5 | 7991.5 | +1104.5 |

Cost is regret against an oracle: `FP × margin + FN × loss`. Declining every
applicant costs **5841**; approving every applicant costs **12442**. A model
is worth deploying only if it beats both.

---

## 🔍 The findings

### 1. A model with AUC 0.72 that is worth exactly nothing

The delinquency rule is a lookup table on one column — the September
repayment status, the thing a credit officer reads off the account screen.
It scores **AUC 0.7205**, which sounds like a working model.

Priced as a decision it saves **zero**. At this cost ratio its cheapest
policy is to decline every applicant, which requires no model at all. The
same is true of the base rate, at AUC 0.50. **Two models, 0.22 of AUC apart,
are worth identically nothing**, and no amount of staring at the AUC column
would tell you that.

Between AUC 0.738 and 0.788 — five points — the money saved multiplies by
five and a half. The relationship between ranking quality and value is not
linear, not monotone at the bottom, and not readable off the metric.

### 2. The threshold is a bigger decision than the model

The same fitted gradient-boosting model:

| Policy | Threshold | Declined | Defaults caught | Cost |
|---|---|---|---|---|
| Cost-optimal | 0.102 | 63.8% | 88.7% | **4712.5** |
| The default 0.5 | 0.500 | 12.7% | 38.8% | 7933.0 |
| Decline everyone | — | 100% | 100% | 5841.0 |

Deployed at 0.5 the model costs **68% more** than the same model at its own
threshold — and **loses to lending to nobody**. Choosing the model bought
1128 units. Choosing the threshold badly gives back 3220.

This does not depend on believing any particular cost ratio:

| R = cost of a missed default | Optimal threshold | Penalty for cutting at 0.5 |
|---|---|---|
| 1 | 0.500 | +0.5% |
| 2 | 0.333 | +4.9% |
| 5 | 0.167 | +33.6% |
| 7.5 | 0.118 | +69.3% |
| 10 | 0.091 | +107.8% |
| 20 | 0.048 | +261.3% |

At R = 1 there is no penalty, because R = 1 *is* the assumption a 0.5 cutoff
encodes. Everywhere a lender would actually live, it is wrong.

One consequence worth stating plainly: **the cost-optimal policy has an
accuracy of 53.4%** — worse than approving everyone, which scores 77.9%.
Accuracy and money point in opposite directions here, and only one of them
is the deliverable.

### 3. `class_weight="balanced"` does nothing, and looks like it does a lot

The reflex fix for a 22%-positive dataset. Against the plain logistic
regression, on the held-out split:

| | Difference | 95% CI | Verdict |
|---|---|---|---|
| AUC | −0.0007 | [−0.0015, +0.0002] | null |
| Regret per account | +0.0034 | [−0.0009, +0.0083] | null |
| McNemar on decisions | — | — | p = 0.38, null |

Every difference that matters is zero. But at a 0.5 cutoff the balanced
model costs 5832 against the plain model's 9598 — it looks **39% better**.
It is not better. The weighting moved the probabilities so that 0.5 landed
in a less absurd place, at the price of a calibration error four times
larger (ECE 0.2379 against 0.0535).

If you compare models at a fixed 0.5 threshold, this is the kind of thing
you will find, and you will be wrong.

### 4. Ranking by probability is the wrong way to spend review capacity

A review team can look at K accounts. Rank by probability of default, or by
probability × balance at risk?

| Capacity | By probability | By expected loss | |
|---|---|---|---|
| 1% (75 accounts) | 63 caught, NT$3.8M | **48 caught, NT$12.8M** | **+234%** |
| 5% (375) | 291 caught, NT$17.3M | 214 caught, NT$34.0M | +96% |
| 10% (750) | 536 caught, NT$32.8M | 372 caught, NT$47.1M | +44% |
| 20% (1,500) | 860 caught, NT$44.5M | 618 caught, NT$61.1M | +37% |

At 1% capacity the expected-loss ordering catches **fifteen fewer defaults
and prevents three times the money**. Recall prefers the losing policy.
Precision prefers the losing policy. AUC prefers the losing policy. All
three are indifferent to the fact that the 300 largest accounts hold more
exposure than the 15,000 smallest.

### 5. The disparities are real, and the reflex fix is the expensive one

One threshold applied to everyone, on a model that ranks almost equally well
inside every group:

| Attribute | AUC gap | Decline-rate gap | Recall gap | FPR gap |
|---|---|---|---|---|
| Sex | 0.0047 | 0.0734 | 0.0286 | 0.0694 |
| Age band | 0.0106 | 0.1374 | 0.0768 | 0.1366 |
| Education | 0.0177 | 0.1663 | 0.0955 | 0.1654 |

The ranking is even; the error rates are not, because the groups default at
different rates before the model exists. Two interventions, both measured:

| Intervention | Cost | Decline gap closed |
|---|---|---|
| Drop sex, age, education, marital status | **+5.3%** | sex 78%, age 44%, **education 19%** |
| Per-group threshold, equal recall | +0.14% to +3.05% | 98–99% of the targeted gap |
| Per-group threshold, equal decline rate | +0.58% to +1.73% | 95–99% of the targeted gap |

**Deleting the protected attributes is both the more expensive intervention
and the less effective one.** It costs about 5% of the model's value
(regret per account +0.0330, CI [+0.0139, +0.0517] — excludes zero) and
closes only a fifth of the education gap, because six months of payment
history proxies for education perfectly well. Fairness through unawareness
is not free and does not work.

Two things this does *not* settle. Per-group thresholds mean explicitly
pricing a protected attribute into a lending decision, which is illegal in
most jurisdictions regardless of what it costs — the number above is what
the law is buying, not an argument against the law. And the definitions
still disagree with each other: equalising recall leaves a decline-rate gap
of 0.057 on education, and equalising decline rates leaves a recall gap of
0.020. That disagreement is a theorem
([Kleinberg et al. 2016](https://arxiv.org/abs/1609.05807);
[Chouldechova 2017](https://arxiv.org/abs/1703.00056)), not a bug to fix.

---

## What this project actually argues

A held-out AUC is not a result. Three things have to be reported with it:

1. **What the decision costs** — with the cost ratio stated as an
   assumption and the conclusion shown across a range of it, because the
   ratio is the one input a reader will want to replace.
2. **Whether the model beats doing nothing** — measured against both naive
   policies, not against a worse model.
3. **Who the errors land on, and what the fix costs** — measured, because
   the intuitive fix here is the wrong one.

---

## 🛠️ Tech Stack

| Component | Tool | Reason |
|---|---|---|
| **Cost model** | Custom (`src/evaluation/decision.py`) | The threshold `1/(1+R)` falls out of the cost ratio; everything else follows from it |
| **Baselines** | Base rate, one-column lookup on `PAY_0` | The decision a credit officer makes with no model |
| **Linear model** | scikit-learn logistic regression | The cheap option, and the reference for the imbalance experiment |
| **Trees** | Random forest, `HistGradientBoostingClassifier` | The expensive option, which has to justify itself in money |
| **Calibration** | Isotonic, cross-validated inside train | A threshold is applied to a probability, so the probability has to mean something |
| **Significance** | Paired bootstrap + exact McNemar | Two of the differences here are null, and saying so requires a test |
| **Fairness** | Custom (`src/evaluation/fairness.py`) | Four definitions, priced against each other |
| **Dashboard** | Streamlit + Plotly | Move the cost assumption and watch every number move |

Gradient boosting is scikit-learn's `HistGradientBoostingClassifier` rather
than LightGBM or XGBoost: it is the same histogram-boosting algorithm and it
drops a compiled OpenMP dependency that makes the other two awkward to
install on macOS. Nothing here turns on which of the three is used — and the
forest and the boosting model are statistically indistinguishable anyway
(AUC difference −0.0044, CI [−0.0097, +0.0007]).

---

## 📁 Repository Layout

```
project-05-credit-risk/
├── src/
│   ├── data/
│   │   ├── loader.py           Download, clean, stratified 60/15/25 split
│   │   └── schema.py           The three gaps in the codebook, decided in one place
│   ├── models/
│   │   ├── baselines.py        Base rate and the one-column delinquency rule
│   │   ├── train.py            The model ladder
│   │   └── run_training.py     Tunes and selects on validation. Never reads test
│   └── evaluation/
│       ├── metrics.py          Discrimination and calibration, kept apart
│       ├── decision.py         The cost model, thresholds, capacity policies
│       ├── fairness.py         Four definitions, and what equalising costs
│       ├── significance.py     Paired bootstrap and exact McNemar
│       └── run_eval.py         The single pass over the test split
├── notebooks/
│   ├── 01_eda.ipynb            The data, the codebook gaps, the exposure skew
│   └── 02_results.ipynb        The ladder, the threshold, the audit
├── tests/                      Split contracts, cost arithmetic, audit arithmetic
├── app.py                      Streamlit dashboard
├── Dockerfile                  Serves the dashboard
└── MODEL_CARD.md               What this model must not be used for
```

---

## 🚀 Setup

```bash
cd project-05-credit-risk
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The dataset (5 MB) downloads itself on first use:

```bash
python -m src.data.loader        # prints the splits and the subgroup cells
```

### Reproducing the results

```bash
python -m src.models.run_training    # grids + selection on validation, ~1 min
python -m src.evaluation.run_eval    # the one pass over test, ~1 min
pytest tests/ -q                     # 35 tests
```

`run_training.py` writes `output/reports/validation.json` — the last place
any choice is made. `run_eval.py` reads it, reads test once, and writes
`test_results.json`, `fairness.json`, and `test_scores.csv.gz`.

### Dashboard

```bash
streamlit run app.py
```

The cost ratio is a slider. Move it and the threshold, the approval rate,
the money and the disparities all move — the argument of the project made
operable rather than asserted.

Or via Docker:

```bash
docker build -t credit-decisions .
docker run -p 8501:8501 credit-decisions
```

No volume mount and no dataset: the dashboard recomputes everything from
`output/reports/test_scores.csv.gz` (201 KB, the 7,500 held-out
predictions), which is committed. It needs neither scikit-learn nor a
fitted model.

### Tests

```bash
pytest tests/ -q
```

35 tests. The split contracts (disjointness, base-rate drift, the
undocumented category codes) and the cost arithmetic, checked against
hand-computed confusion matrices rather than trusted — including the case
that equalising a group rate can come out *cheaper* than the cost-optimal
single threshold, which is true and was initially asserted to be impossible.

---

## ⚠️ Limitations

Full writeup in [`MODEL_CARD.md`](MODEL_CARD.md). The short version:

- **The cost ratio is an assumption, not a measurement.** R = 7.5 stands in
  for a ~10% net margin and a ~75% loss given default. A lender would
  replace it with their own; every result is also reported as a curve over R
  so the conclusions do not rest on the number.
- **One market, one moment.** Taiwan, 2005, during a card-lending crisis
  that produced a 22% default rate. That base rate is far above a normal
  portfolio and it moves every threshold in this project.
- **The model must not be deployed as a lending decision.** Its
  cost-optimal policy declines 64% of applicants, and it exhibits measured
  disparities across two attributes protected under consumer-credit law.
- **Three undocumented coding gaps** in `EDUCATION`, `MARRIAGE` and the
  repayment-status columns are resolved by judgment, documented in
  `src/data/schema.py`.

---

## Author

**Manuel Corona**
