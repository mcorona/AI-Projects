# Model Card — Credit Decisions, Priced

## Model details

**What it is.** A gradient-boosting classifier (scikit-learn
`HistGradientBoostingClassifier`, `learning_rate=0.06`, `max_leaf_nodes=31`)
that estimates the probability a credit-card account defaults in the
following month, plus a decision layer that turns that probability into an
approve/decline recommendation given an explicit cost ratio.

**Trained on.** 18,000 accounts from the UCI "default of credit card
clients" dataset — Taiwanese accounts observed April–September 2005, with
the October 2005 outcome. 4,500 accounts held out for tuning and threshold
selection; 7,500 read exactly once, at the end.

**Inputs.** 23 features: credit limit, age, sex, education, marital status,
six months of repayment status, six months of bill amounts, six months of
payment amounts.

**Output.** A probability in [0, 1], and a decision under a stated cost
ratio R (default 7.5, giving a decline threshold of 0.118; the threshold
frozen on validation is 0.102).

**Author.** Manuel Corona. **Date.** August 2026.

---

## Intended use

**This model is a demonstration of evaluation method. It is not a lending
system and must not be used as one.**

It exists to show what changes when a classifier is scored by the cost of
its decisions rather than by AUC. The reusable parts are
`src/evaluation/decision.py` and `src/evaluation/fairness.py` — the cost
model and the audit — not the fitted estimator.

Appropriate uses: studying cost-sensitive thresholding, reproducing the
results, adapting the cost and audit code to a real portfolio with real
cost parameters.

---

## Out of scope

**Do not use this model to decide who gets credit.** Four independent
reasons, any one of which is sufficient:

1. **It is trained on one market at one moment.** Taiwan, 2005, during a
   card-lending crisis. The 22.12% default rate is several times a normal
   portfolio's, and every threshold in this project is computed against
   that base rate. Applied to a 3%-default portfolio, the cost-optimal
   threshold and every number that follows from it are wrong.

2. **It exhibits measured disparities on protected attributes.** At the
   deployed threshold, men are declined 7.3 percentage points more often
   than women and applicants over 50 are declined 13.7 points more often
   than those aged 30–39. These follow from real base-rate differences and
   from a model that ranks almost equally well in every group — which does
   not make them lawful. Sex and age are protected under consumer-credit
   law in most jurisdictions (in the US, ECOA covers both).

3. **The cost ratio is assumed, not measured.** R = 7.5 stands in for a
   ~10% net margin and a ~75% loss given default. It is a plausible
   textbook figure, not this portfolio's actual economics.

4. **The cost-optimal policy declines 63.8% of applicants.** That is what
   minimises cost on this population at this ratio. It is not a business.

**Also out of scope:** any use of the per-group thresholds computed in the
fairness audit. They are priced there to show what the legal constraint
costs, and pricing a protected attribute into a credit decision is
prohibited in most jurisdictions. The audit measures the trade; it does not
recommend taking it.

---

## Performance

Held-out test split, 7,500 accounts, 1,659 defaults, NT$380.5M at risk.

| Metric | Value |
|---|---|
| ROC AUC | 0.7881 |
| PR AUC | 0.5684 |
| ECE (10 equal-width bins) | 0.0115 |
| Brier score | 0.1327 |
| Cost at the frozen threshold (0.102) | 4712.5 |
| Cost of declining everyone | 5841.0 |
| Cost of approving everyone | 12442.5 |
| Share of the oracle gap closed | 19.3% |

At the frozen threshold: declines 63.8% of applicants, catches 88.7% of
defaults, declines 56.7% of good customers, precision 30.8%, **accuracy
53.4%** — lower than approving everyone (77.9%). Accuracy is not the
objective and is reported here only so the discrepancy is visible.

### Subgroup performance

| Attribute | Group | n | Default rate | Declined | Recall | AUC |
|---|---|---|---|---|---|---|
| Sex | Male | 3,017 | 0.2463 | 0.6815 | 0.9031 | 0.7841 |
| Sex | Female | 4,483 | 0.2043 | 0.6081 | 0.8745 | 0.7888 |
| Age | 21–29 | 2,420 | 0.2202 | 0.6645 | 0.9006 | 0.7881 |
| Age | 30–39 | 2,791 | 0.2082 | 0.5926 | 0.8640 | 0.7885 |
| Age | 40–49 | 1,626 | 0.2312 | 0.6371 | 0.8803 | 0.7829 |
| Age | 50+ | 663 | 0.2549 | 0.7300 | 0.9408 | 0.7935 |
| Education | Graduate school | 2,661 | 0.1905 | 0.5415 | 0.8245 | 0.7838 |
| Education | University | 3,474 | 0.2375 | 0.6983 | 0.9200 | 0.7875 |
| Education | High school | 1,246 | 0.2592 | 0.7079 | 0.9009 | 0.7698 |

The education "other/unknown" cell — 119 accounts, 4 defaults — is excluded
from every reported disparity. Rates estimated from four events are not
rates. It is carried through the pipeline and is visible in
`output/reports/fairness.json` marked `reliable: false`.

---

## Data and its defects

**Three gaps in the published codebook**, each resolved by judgment in
`src/data/schema.py`:

- `EDUCATION` contains codes 0, 5 and 6 (345 rows) that the codebook does
  not define. Folded into 4, "others", rather than dropped — deleting rows
  on the basis of an administrative coding gap silently removes a
  subpopulation.
- `MARRIAGE` contains code 0 (54 rows), undefined. Folded into 3, "others".
- The repayment-status columns contain −2 and 0, undefined. Read as "no
  balance" and "revolving credit, minimum paid", consistent with the
  billing columns. This matters: those two codes are 77.3% of the column,
  and treating the column as a clean ordinal scale asserts that −2 is
  "better" than −1, which is a claim about no-debt versus debt-repaid that
  the data does not support. Accounts coded 0 in fact default *less* than
  accounts coded −1 (0.128 against 0.168), despite carrying eighteen times
  the median balance.

**No leakage.** All features are observations from April–September 2005;
the target is the October 2005 outcome. The most predictive single feature,
`PAY_0`, is the September delinquency status, which is available at
decision time.

**The column named `PAY_0`** is the September repayment status; there is no
`PAY_1`. That is an error in the original upload, preserved so that results
stay comparable with published work on this dataset.

---

## Ethical considerations

The fairness section of this project reports something uncomfortable in
both directions, and neither half should be quoted without the other.

**Dropping the protected attributes does not fix the disparity.** It closes
78% of the decline-rate gap on sex but only 19% on education, because six
months of payment history proxies for education. It also costs about 5% of
the model's value (regret per account +0.0330, 95% CI [+0.0139, +0.0517]).
"Fairness through unawareness" is measurably neither free nor effective.

**Per-group thresholds are cheap and effective, and mostly illegal.**
Equalising recall across groups costs 0.14%–3.05% of the model's value and
closes 98–99% of the targeted gap. That is a small number, and it is the
price of a constraint that exists for reasons this project does not
measure.

**The fairness definitions disagree, and that is a theorem.** Equalising
recall on education leaves a decline-rate gap of 0.057; equalising decline
rates leaves a recall gap of 0.020. With unequal base rates and an
imperfect model, demographic parity, equal opportunity and predictive
parity cannot hold simultaneously (Kleinberg et al. 2016; Chouldechova
2017). This project prices the trade; it does not resolve it, and no
project can.

**The base rates themselves are not neutral.** Men default more than women
in this data and applicants over 50 default more than those in their
thirties. Those are facts about a specific credit market in 2005, shaped by
who was lent to and on what terms. A model trained on them reproduces the
consequences of that lending, and calling the result "accurate" does not
make it a fair basis for future lending.

---

## Reproducibility

Seeds are fixed (`SPLIT_SEED = 20260827` for the split, `SEED = 20260827`
for the models and the bootstrap). The split is deterministic and tested to
be so. The dataset downloads itself from UCI.

```bash
python -m src.models.run_training   # validation.json -- all selection happens here
python -m src.evaluation.run_eval   # the single test pass
pytest tests/ -q                    # 35 tests
```

Known non-determinism: none observed. `RandomForestClassifier` and
`HistGradientBoostingClassifier` are seeded; the paired bootstrap is seeded.

---

## Contact

**Manuel Corona**
