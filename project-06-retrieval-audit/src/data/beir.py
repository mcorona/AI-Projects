"""
BEIR datasets: download, parse, and describe.

This is Project 3's FiQA loader generalized to the seven corpora this audit
runs over. Nothing about the format changes between BEIR datasets -- a
corpus.jsonl, a queries.jsonl, and a qrels TSV per split -- so the only
per-dataset facts worth recording are the ones that affect how a result
should be read.

The datasets were chosen before any of them was run, to span the range
where the hypothesis under test should break: corpora where BM25 is known
to be strong (SciFact, TREC-COVID), one where dense retrieval is known to
struggle for structural reasons (ArguAna, whose "queries" are whole
arguments rather than questions), and FiQA as the anchor, because Project 3
already measured it and this audit has to be able to contradict that.

Author: Manuel Corona
"""

import csv
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.request import urlretrieve

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class Dataset:
    name: str          # BEIR archive name
    split: str         # which qrels split carries the judgments
    label: str         # human-readable
    domain: str
    note: str          # why it is in the sample


# Seven corpora, fixed in advance. Sizes are the published BEIR figures and
# are asserted against the parsed data in tests.
DATASETS: List[Dataset] = [
    Dataset("scifact", "test", "SciFact", "Scientific claims",
            "BM25 is a strong baseline here; short, technical, high lexical overlap."),
    Dataset("nfcorpus", "test", "NFCorpus", "Medical / nutrition",
            "Small corpus, long queries, heavy vocabulary mismatch."),
    Dataset("arguana", "test", "ArguAna", "Argument retrieval",
            "Queries are whole arguments and the target is a counter-argument, "
            "so lexical overlap actively misleads. The structural stress case."),
    Dataset("scidocs", "test", "SciDocs", "Scientific citation",
            "Citation prediction from a title -- neither retriever's native task."),
    Dataset("trec-covid", "test", "TREC-COVID", "Biomedical",
            "50 queries only, deeply judged. BM25 historically competitive."),
    Dataset("quora", "test", "Quora", "Duplicate questions",
            "Question-to-question; dense retrieval's best case."),
    Dataset("fiqa", "test", "FiQA-2018", "Financial QA",
            "The anchor. Project 3 measured RRF fusion costing 4.7 nDCG here, "
            "and this audit exists partly to test whether that generalizes."),
]

BY_NAME: Dict[str, Dataset] = {d.name: d for d in DATASETS}


def download(name: str, force: bool = False) -> Path:
    """Fetch and unzip one BEIR dataset into data/raw/<name>/."""
    target = RAW_DIR / name
    if target.exists() and not force:
        return target
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"{name}.zip"
    if not zip_path.exists() or force:
        url = f"{BEIR_BASE}/{name}.zip"
        print(f"  downloading {url}")
        urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink(missing_ok=True)
    if not target.exists():
        raise FileNotFoundError(f"{name}.zip did not contain a {name}/ directory")
    return target


def load_corpus(name: str) -> Dict[str, str]:
    """
    doc_id -> text.

    Title and body are concatenated, which is what the BEIR reference
    implementation does. Doing anything else here would silently make the
    numbers incomparable to published results, and comparability to
    published results is what makes this an audit rather than an opinion.
    """
    path = download(name) / "corpus.jsonl"
    corpus: Dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            corpus[rec["_id"]] = f"{title} {text}".strip()
    return corpus


def load_queries(name: str) -> Dict[str, str]:
    path = download(name) / "queries.jsonl"
    with path.open(encoding="utf-8") as fh:
        return {r["_id"]: r["text"] for r in map(json.loads, fh)}


def load_qrels(name: str, split: str = "test") -> Dict[str, Dict[str, int]]:
    """query_id -> {doc_id: graded relevance}. Zero-relevance rows dropped."""
    path = download(name) / "qrels" / f"{split}.tsv"
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        if header and header[0].lower() not in ("query-id", "query_id"):
            fh.seek(0)
            reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = row[0], row[1], int(float(row[2]))
            if score > 0:
                qrels[qid][did] = score
    return dict(qrels)


def load(name: str, split: str = "test"):
    """
    Returns (corpus, queries, qrels) with queries restricted to those that
    actually carry judgments -- scoring a query with no qrels contributes a
    zero to the mean and quietly deflates every system equally.
    """
    corpus = load_corpus(name)
    qrels = load_qrels(name, split)
    queries = {q: t for q, t in load_queries(name).items() if q in qrels}
    return corpus, queries, qrels


def describe(name: str, split: str = "test") -> Dict:
    corpus, queries, qrels = load(name, split)
    judged = [len(v) for v in qrels.values()]
    return {
        "dataset": name,
        "documents": len(corpus),
        "queries_with_qrels": len(queries),
        "judgments": int(sum(judged)),
        "mean_judged_per_query": round(sum(judged) / len(judged), 2) if judged else 0.0,
        "mean_query_words": round(
            sum(len(t.split()) for t in queries.values()) / len(queries), 1) if queries else 0.0,
        "mean_doc_words": round(
            sum(len(t.split()) for t in corpus.values()) / len(corpus), 1) if corpus else 0.0,
    }


if __name__ == "__main__":
    rows = []
    for d in DATASETS:
        print(f"{d.label} ...")
        rows.append((d, describe(d.name, d.split)))
    print()
    hdr = f"{'dataset':<12} {'domain':<22} {'docs':>9} {'queries':>8} {'judg/q':>7} {'q words':>8} {'d words':>8}"
    print(hdr)
    print("-" * len(hdr))
    for d, s in rows:
        print(f"{d.label:<12} {d.domain:<22} {s['documents']:>9,} "
              f"{s['queries_with_qrels']:>8,} {s['mean_judged_per_query']:>7.1f} "
              f"{s['mean_query_words']:>8.1f} {s['mean_doc_words']:>8.1f}")
