# Model Card — Financial RAG

A retrieval-augmented question answering system over financial forum text,
built as an evaluation exercise rather than a product.

---

## System description

| | |
|---|---|
| **Task** | Answer personal-finance questions from a fixed document corpus, with citations |
| **Corpus** | FiQA-2018 (BEIR distribution) — 57,638 financial forum answers |
| **Retriever** | `BAAI/bge-base-en-v1.5` bi-encoder, exhaustive cosine search |
| **Generator** | `claude-opus-5`, structured output (answer / citations / context_sufficient) |
| **Passages per answer** | 5, truncated to 1,200 characters each |
| **Judge** | `claude-opus-5`, cross-checked against `claude-haiku-4-5` |

### Intended use

Demonstrating and measuring a RAG pipeline: which retrieval components earn
their cost, how much retrieval quality survives into answer quality, and how
to calibrate an LLM judge before believing it.

### The result this card exists to qualify

On this benchmark, the full RAG pipeline answers **fewer** questions
correctly than the same model with no retrieval at all: 71.3% against
82.0%, exact McNemar p = 0.020 over the same 150 questions.

The system is not broken. Restricted to questions it did not abstain on, it
is *more* accurate than closed-book (86.8% vs 82.0%) and, given the gold
passages, correct every time (100%, n=127). It loses the aggregate because
it declines to answer when its passages do not support one, and abstentions
score as not-correct.

That is the trade this system makes and it should be stated plainly: **it
exchanges raw answer rate for attributability.** For a domain the generator
has already memorised — retail personal finance is exactly that — the trade
is a bad deal. For proprietary documents, post-cutoff facts, or any setting
where an answer must be traceable to a source, it is the whole point.
Anyone reusing this pipeline should measure which situation they are in
rather than assuming.

### Out of scope

- **Financial or investment advice.** The corpus is anonymous forum opinion,
  much of it years old, none of it verified. The system retrieves and
  summarizes it; it does not fact-check it, date it, or assess the
  competence of whoever wrote it.
- **Any automated decision affecting a person** — credit, eligibility,
  advice served without a human reading it.
- **Domains other than English-language retail personal finance.** Not
  tested on filings, research notes, regulatory text, or non-English text.
- **Questions whose answer is not in the corpus.** By design the system
  abstains rather than guessing, but abstention is not perfect (below).

---

## How it was evaluated

Two phases, both against human ground truth where ground truth exists.

**Retrieval** — nine configurations scored on 648 test queries against
1,706 human relevance judgments, using nDCG@10, Recall@k and MRR@10
implemented in `src/evaluation/ir_metrics.py` with hand-computed unit tests.
Differences are tested with a paired bootstrap (10,000 resamples).

**Generation** — 150 test questions under four conditions (no retrieval /
BM25 passages / bge passages / gold passages). Three of the metrics need no
model judgment at all: abstention rate, citation precision against the
qrels, and invalid-citation rate. The two that do — groundedness and
correctness — are produced by an LLM judge whose reliability is measured
before its verdicts are used, with the caveat in limitation 4 below.

### Validation against published numbers

The retrieval harness reproduces BEIR's published FiQA results to within
0.002 nDCG@10 for both BM25 (0.2374 vs 0.236) and bge-base-en-v1.5 (0.4062
vs 0.406). A harness that is subtly wrong still produces plausible numbers,
so this check is the precondition for everything else in the project.

---

## What it does well

- **Detects its own retrieval failures.** Splitting the best RAG condition
  by whether the retriever actually surfaced a gold document: it abstains
  5.6% of the time when it did, and 39.3% when it did not — seven times
  more often, on exactly the questions where abstention is correct. On the
  same split, correctness is 92.1% vs 41.0%. The abstention is targeted,
  not blanket caution.
- **Is accurate when it does answer.** 86.8% correct on non-abstained
  questions, above the 82.0% closed-book rate, and 100% given gold
  passages.
- **Never invents a citation.** Across 600 generated answers, zero cited a
  passage number that was not in its context.
- **Cites relevant passages more often when given better ones.** Citation
  precision against human judgments: 0.199 (BM25) → 0.303 (bge) → 1.000
  (gold).

---

## What it does badly, and known limitations

### 1. The best retriever misses entirely on ~20% of questions

`bge-base-en-v1.5` returns nothing relevant in its top 10 for about a fifth
of test questions. On those, no generator can produce a grounded answer;
the correct behaviour is abstention, and the system does not always take it.

### 2. Even gold passages are insufficient 15.3% of the time

Handed the documents human annotators marked relevant, the system still
reports that it cannot answer in roughly one case in seven. That is a
property of FiQA — a document can be topically relevant without containing
the answer — and it bounds every downstream number. No amount of retrieval
engineering moves it.

### 3. The qrels are sparse

FiQA marks 2.63 documents per question as relevant out of 57,638. Documents
that are genuinely relevant but unjudged count as misses. This understates
every retriever, and it understates a *good* retriever more than a bad one,
because a good retriever surfaces more unjudged-but-relevant material.
Absolute nDCG values should be read as "on this benchmark", not "in
reality".

### 4. The judge is a Claude model judging a Claude model

The generator and the primary judge are both `claude-opus-5`. This is a
real self-preference risk and cannot be argued away, only measured. Two
checks were designed; **only one of them actually ran.**

- **Self-consistency — ran.** The same judge re-judged 297 items and agreed
  with itself on 96.6% of exact verdicts (98.0% on the correct / not-correct
  collapse). The judge is precise enough to resolve the 11-point gap that
  carries the headline finding.
- **Cross-model agreement — did NOT run.** The account's API credit balance
  was exhausted before this batch was submitted. It is recorded as
  `"status": "not run"` in `output/reports/generation_metrics.json` rather
  than omitted.

**Consequence: every judged number in this project is un-cross-validated.**
A judge that systematically favours its own family's output would inflate
all four conditions, but not necessarily equally, and nothing here rules
that out. The check costs roughly $0.40 to run (`--phase judge` resumes and
submits only the missing batch).

The three judge-free metrics — abstention rate, citation precision against
the qrels, and invalid-citation rate — involve no model opinion and are
unaffected.

Neither check is a substitute for human labels, which this project does not
have. The judged numbers should be read as ordinal (condition A beat
condition B) rather than absolute (X% of answers are correct).

### 5. Conclusions are domain-bound

"Hybrid fusion and reranking hurt" is a finding about *this corpus* with
*these components*. RRF weights every input system equally by rank; fusing
a retriever at nDCG@10 = 0.237 with one at 0.406 drags the strong one down
by construction, and a weighted fusion tuned on a dev split might not.
Likewise both rerankers were trained on general web retrieval, and a
reranker fine-tuned on financial text could plausibly beat the bi-encoder.
Neither was tested.

### 6. Truncation is a confound in the MiniLM comparison

`all-MiniLM-L6-v2` caps at 256 tokens and silently discards 19.8% of all
corpus tokens; `bge-base-en-v1.5` caps at 512 and discards 6.0%. bge wins
by 3.8 nDCG points, but part of that is reading more of each document
rather than reading it better. The two effects are not separated here.

### 7. No chunking

Documents are embedded whole and truncated at the encoder limit rather than
split into overlapping chunks. For a corpus with a 90-word median document
this is defensible; for one with long documents it would be the first thing
to fix, and the 4.1% of documents that exceed bge's limit are the ones this
system handles worst.

### 8. Cost and latency are not production numbers

The evaluation runs through the Batches API at 50% pricing with no latency
requirement. A live system paying standard pricing with a p95 latency
target would make different choices — including, plausibly, a smaller
generator.

---

## Reproducing

```bash
pip install -r requirements.txt
python -m src.evaluation.run_retrieval_eval      # no API key, no cost
python -m src.evaluation.significance
python -m src.evaluation.run_generation_eval --phase pilot --n 3   # price it first
```

The FiQA download, corpus embeddings, ranked runs, and batch ids are all
cached under `data/processed/`, so every stage is restartable and no step
is paid for twice. Full results: `output/reports/`.

---

## Author

**Manuel Corona**
