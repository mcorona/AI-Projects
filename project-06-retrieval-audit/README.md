# When Hybrid Retrieval Hurts

## An audit of a default

This is not a portfolio project. It is an audit of a recommendation that
retrieval tooling ships as a default and that practitioner guidance repeats
without conditions.

**The claim under audit**, quoted from 2026 practitioner guidance:

> "Hybrid search consistently outperforms either method alone."
>
> "If your RAG system uses pure vector search, adding BM25 is the single
> highest-impact retrieval upgrade you can make."
>
> "Start with RRF at k=60 as the zero-config default."

**The finding:** both halves of that advice are true on one side of a
boundary the guidance never mentions, and false on the other. Fusion and
reranking each help a first stage that is not already strong, and each
degrade one that is. Applying both to a modern dense retriever produced a
**significantly worse** system than applying neither, on all four corpora
tested.

The fusion half in detail: the claim is true, and it is true only on one side
of a boundary the guidance never mentions. Fusing BM25 into a dense retriever
helps when that retriever is not already beating BM25, and hurts — by more
than the fusion ever gains — when it is. On four BEIR corpora with two
dense retrievers of different strength, the correlation between a
retriever's existing advantage over BM25 and the benefit it gets from
fusion is **−0.913**.

---

## Results

Eight (corpus, retriever) pairs. Every configuration at RRF k=60, top-100,
paired bootstrap over queries.

| Corpus | Dense retriever | Dense alone | + BM25 (RRF) | Δ | p | Gap over BM25 |
|---|---|---:|---:|---:|---:|---:|
| SciFact | MiniLM | 0.6451 | 0.6943 | **+0.0493** | 0.0006 | −0.0106 |
| NFCorpus | MiniLM | 0.3165 | 0.3364 | **+0.0200** | 0.0166 | +0.0080 |
| NFCorpus | bge-base | 0.3743 | 0.3550 | **−0.0193** | 0.0110 | +0.0658 |
| SciFact | bge-base | 0.7404 | 0.7049 | **−0.0355** | 0.0102 | +0.0847 |
| ArguAna | MiniLM | 0.5024 | 0.4998 | −0.0026 | 0.7200 | +0.0892 |
| FiQA | MiniLM | 0.3687 | 0.3601 | −0.0086 | 0.4002 | +0.1313 |
| FiQA | bge-base | 0.4062 | 0.3588 | **−0.0474** | <0.0001 | +0.1689 |
| ArguAna | bge-base | 0.6388 | 0.5591 | **−0.0798** | <0.0001 | +0.2257 |

Rows sorted by the dense retriever's standalone advantage over BM25. nDCG@10
throughout. Bold = significant at p < 0.05.

**Fusion never helps once the dense retriever leads BM25 by more than about
one point of nDCG.** It helps materially only in the two rows where the
dense retriever is at or below BM25. In between there is a band where it
does nothing measurable, and past it the loss grows to eight points.

### The within-corpus control

The cleanest evidence is not the trend across corpora — corpora differ in
many ways — but the sign flip *inside* one. On SciFact, holding the corpus,
the BM25 run and the fusion method fixed, and changing only which dense
retriever is fused:

- MiniLM (0.6451, below BM25's 0.6557): fusion **+0.0493**, p = 0.0006
- bge-base (0.7404, above BM25): fusion **−0.0355**, p = 0.0102

Same operation, same data, opposite and significant directions. NFCorpus
reproduces it. That is a mechanism, not a corpus idiosyncrasy.

### The rule this suggests

Run the two retrievers separately before deciding to fuse. If the dense run
does not clearly beat BM25 on your data, fusion is likely to help. If it
beats BM25 by more than a point or two of nDCG, fusion is likely to cost
you — and the stronger your retriever, the more it costs.

This is a decision rule the guidance does not offer, and it requires no
labelled data beyond what evaluating either retriever already requires.

---

## Validation

An audit that reports someone else's defaults are wrong has to first show
its own instrument is right.

| Corpus | System | Observed | Published | Δ | Status |
|---|---|---:|---:|---:|---|
| SciFact | MiniLM | 0.6451 | 0.645 | +0.0001 | ok |
| SciFact | bge-base | 0.7404 | 0.742 | −0.0016 | ok |
| NFCorpus | MiniLM | 0.3165 | 0.318 | −0.0015 | ok |
| NFCorpus | bge-base | 0.3743 | 0.373 | +0.0013 | ok |
| ArguAna | MiniLM | 0.5024 | 0.501 | +0.0014 | ok |
| ArguAna | bge-base | 0.6388 | 0.636 | +0.0028 | ok |
| FiQA | MiniLM | 0.3687 | 0.369 | −0.0003 | ok |
| FiQA | bge-base | 0.4062 | 0.406 | +0.0002 | ok |

Eight dense checks, maximum deviation **0.0028**.

BM25 is a range check rather than a match check: this pipeline uses
`rank_bm25`, the BEIR paper used Elasticsearch, and Anserini is a third
implementation, with legitimate spread between them. Observed −0.0093
(SciFact), −0.0165 (NFCorpus), +0.0014 (FiQA), and +0.0981 on ArguAna,
where the BEIR paper reports 0.315 and Anserini around 0.397 against this
pipeline's 0.4131. **BM25 is half of every fusion measured here, so any bias
in it propagates into the headline result.** It is declared rather than
buried.

> ⚠️ **The published reference values are not yet verified.** They are
> transcribed from recollection of the BEIR paper and the MTEB leaderboard,
> not fetched from a primary source, and every entry is flagged `"recalled"`
> in `src/evaluation/reference.py`. They must be checked against their
> sources before this audit is published anywhere. A validation table built
> on half-remembered numbers is worse than no validation table, because it
> looks like diligence.

### A defect this pipeline had, and how it was caught

The first run of ArguAna produced bge-base at 0.4560 against a published
~0.636. Too large to be tokenization.

Cause: 92% of ArguAna's judged query ids are also document ids, and a
query's own text is never among its relevant documents. A dense retriever
matches it at cosine 1.0 and parks it at rank 1, displacing every real
answer by one position. The BEIR reference implementation excludes it; this
one did not.

It mattered more than a uniform penalty would: a dense model self-matches
perfectly and BM25 only strongly, so the bug penalised precisely the
retrievers being compared — and it made the headline pattern look *cleaner*
than it is. Fixing it removed a bias that favoured this audit's own
hypothesis.

**Nothing caught this except comparing against the published number.** No
test would have. That is the argument for the validation step, and it is
the reason it comes before the results and not after.

---

## Stage 2: the reranker

The other half of the same guidance:

> "Add a cross-encoder reranker after fusion for the biggest single
> precision gain."

Also unconditional. Same four corpora, `ms-marco-MiniLM-L-6-v2` over the
top 50, paired bootstrap.

### The control comes first

A cross-encoder over a weak lexical first stage is the textbook case where
reranking shines. If that does not gain, the reranker is misconfigured and
nothing else in this stage means anything.

| Corpus | BM25 | + reranker | Δ | p |
|---|---:|---:|---:|---:|
| FiQA | 0.2374 | 0.3239 | **+0.0865** | <0.0001 |
| NFCorpus | 0.3085 | 0.3363 | **+0.0278** | <0.0001 |
| SciFact | 0.6557 | 0.6808 | +0.0252 | 0.0970 |
| ArguAna | 0.4131 | 0.4250 | +0.0119 | 0.2048 |

It gains **36% relative** on FiQA, reproducing Project 3's figure exactly.
The reranker works. Note also that the two corpora where it fails to gain
significantly are the two where BM25 is already strong — the same boundary
Stage 1 found.

### And then it degrades the strong first stage, everywhere

| Corpus | bge-base | + reranker | Δ | p |
|---|---:|---:|---:|---:|
| NFCorpus | 0.3743 | 0.3582 | **−0.0161** | 0.0356 |
| FiQA | 0.4062 | 0.3851 | **−0.0212** | 0.0146 |
| SciFact | 0.7404 | 0.7013 | **−0.0391** | 0.0062 |
| ArguAna | 0.6388 | 0.4316 | **−0.2073** | <0.0001 |

Four corpora, four significant losses.

### The full recommended stack against doing neither

Fuse BM25 with the dense retriever, then rerank the result — exactly what
the guidance prescribes — measured against the plain dense retriever it is
supposed to improve:

| Corpus | bge alone | Fused + reranked | Δ | p |
|---|---:|---:|---:|---:|
| NFCorpus | 0.3743 | 0.3583 | **−0.0160** | 0.0486 |
| FiQA | 0.4062 | 0.3743 | **−0.0319** | 0.0008 |
| SciFact | 0.7404 | 0.6958 | **−0.0446** | 0.0032 |
| ArguAna | 0.6388 | 0.4245 | **−0.2143** | <0.0001 |

**Four corpora, four significant losses.** Following both recommendations
produced a worse retriever than following neither, every time, on a modern
dense retriever.

### Two mechanisms, not one

The tidy story would be that everything reduces to first-stage strength.
It mostly does — but ArguAna does not fit, and forcing it in would be
overclaiming.

There, reranking degrades *every* arm it touches, including the weak MiniLM
(−0.0653, p<0.0001), and it destroys bge, costing a third of its score. The
likely reason is structural rather than about strength: ArguAna's task is to
retrieve a *counter*-argument, while a cross-encoder trained on MS MARCO
scores topical relevance. Its notion of a good match is close to the
opposite of the task's. That is a second failure mode — reranker/task
mismatch — and it is not the same as the first.

So: reranking helps a weak first stage on a task its training resembles, and
hurts otherwise. Two conditions, neither of them stated in the guidance.

### The second reranker

`bge-reranker-base` was run on NFCorpus only, to check that a degradation is
not simply one badly chosen model. It hurt bge there too, and harder:
**−0.0473, p<0.0001**, against ms-marco's −0.0161. It was not run on the
other three corpora — it is roughly nine times slower per pair, and one
corpus carrying both rerankers already answers the objection it exists to
answer.


---

## Method

Seven BEIR corpora were selected in advance, with the reason for each
recorded in `src/data/beir.py`, to span the range where the hypothesis
should break. Four were run in this stage:

| Corpus | Domain | Docs | Queries | Words/query |
|---|---|---:|---:|---:|
| SciFact | Scientific claims | 5,183 | 300 | 12.5 |
| NFCorpus | Medical / nutrition | 3,633 | 323 | 3.3 |
| ArguAna | Argument retrieval | 8,674 | 1,406 | 193.6 |
| FiQA-2018 | Financial QA | 57,638 | 648 | 10.9 |

Five arms per corpus: BM25, MiniLM, bge-base-en-v1.5, RRF(BM25, MiniLM),
RRF(BM25, bge). Two dense retrievers of deliberately different strength, so
the question is not only whether fusion helps on average but whether one
corpus can show it helping one retriever and hurting another.

```bash
python -m src.data.beir                  # download + inventory
python -m src.evaluation.run_stage1      # fusion: five arms, four corpora
python -m src.evaluation.reference       # validation against published
python -m src.evaluation.run_stage2      # reranking: four arms, four corpora
python -m src.evaluation.run_stage2 --second-reranker --datasets nfcorpus
```

About 12 minutes for Stage 1 and 25 for Stage 2 on an M-series Mac with
embeddings cached. Stage 2 writes its report after every corpus, so an
interrupt costs the corpus in flight rather than the run.

---

## Limits

- **Four corpora, two retrievers, eight comparisons.** The rule is a
  hypothesis with good support, not an established result. SciDocs,
  TREC-COVID and Quora are downloaded and not yet run.
- **The trend is not clean.** Ordered by gap, two rows invert: fusion hurts
  MiniLM *less* than its gap predicts on ArguAna and FiQA. The identity of
  the dense model appears to matter beyond its score — a weaker model may
  leave more complementary signal for BM25 to contribute even while
  outscoring it. Not investigated here.
- **RRF k=60, equal weights, top-100 only.** Weighted or tuned fusion might
  behave differently, and the guidance recommends convex combination once
  labelled data exists. Untested.
- **Reranking: one cross-encoder on four corpora, a second on one.** Depth
  50, and depth is a parameter that was not varied. A reranker with a
  different training mix might behave differently on ArguAna, where the
  failure looks task-shaped rather than strength-shaped.
- **The reference values are unverified**, as stated above.

---

## Why this exists

Project 3 in this portfolio measured RRF fusion costing 4.7 nDCG points on
FiQA, against guidance saying fusion always helps. One dataset is an
anecdote. This audit exists to find out whether that result generalizes or
whether FiQA was the exception — and was designed so that "FiQA was the
exception" was a publishable outcome.

FiQA reproduces here exactly: BM25 0.2374, bge 0.4062, fusion −0.0474. It
was not the exception. It was one point on a line.

## Author

**Manuel Corona**
