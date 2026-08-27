"""
FiQA-2018 (BEIR) loading: corpus, queries, and relevance judgments.

FiQA is a financial-domain question answering retrieval benchmark: the
questions are real investor questions from StackExchange/Reddit, and the
corpus is financial forum answers. It is the right benchmark for this
project for two reasons:

  1. It ships human relevance judgments (qrels), so retrieval quality is
     measurable against ground truth instead of eyeballed. Most RAG demos
     have no ground truth at all, which is exactly the gap this project
     is about.
  2. It is a published BEIR benchmark, so the numbers here can be sanity
     checked against the literature rather than existing in a vacuum.

Author: Manuel Corona
"""

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from urllib.request import urlretrieve

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FIQA_DIR = RAW_DIR / "fiqa"


def download_fiqa(force: bool = False) -> Path:
    """
    Download and unzip the BEIR FiQA-2018 dataset into data/raw/fiqa/.

    Returns the dataset directory. No-op if it is already present.
    """
    if FIQA_DIR.exists() and not force:
        return FIQA_DIR

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "fiqa.zip"
    if not zip_path.exists() or force:
        print(f"Downloading {BEIR_URL} ...")
        urlretrieve(BEIR_URL, zip_path)
    print(f"Extracting to {RAW_DIR} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    return FIQA_DIR


def load_corpus() -> Dict[str, Dict[str, str]]:
    """
    Load the document corpus.

    Returns:
        {doc_id: {"title": str, "text": str}} -- title is empty for most
        FiQA documents (they are forum answers, not titled articles).
    """
    corpus = {}
    with open(FIQA_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
    return corpus


def load_queries() -> Dict[str, str]:
    """Load all queries (train + dev + test share one file). {query_id: text}."""
    queries = {}
    with open(FIQA_DIR / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels(split: str = "test") -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments for one split.

    Args:
        split: "train", "dev", or "test".

    Returns:
        {query_id: {doc_id: relevance}} for judged pairs only. FiQA uses
        binary relevance (score 1); documents absent from a query's dict
        are treated as non-relevant, which is the standard BEIR
        assumption.
    """
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    path = FIQA_DIR / "qrels" / f"{split}.tsv"
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header[0] != "query-id":  # not a header row after all -- rewind
            f.seek(0)
            reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = row[0], row[1], int(row[2])
            if score > 0:
                qrels[qid][did] = score
    return dict(qrels)


def load_split(split: str = "test"):
    """
    Convenience loader: (corpus, queries_for_split, qrels_for_split).

    Only queries that have at least one judged relevant document are
    returned -- an unjudged query cannot contribute to a retrieval metric,
    and silently scoring it as 0 would deflate every model equally but
    make the numbers incomparable to published BEIR results.
    """
    download_fiqa()
    corpus = load_corpus()
    all_queries = load_queries()
    qrels = load_qrels(split)
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    return corpus, queries, qrels


def doc_text(doc: Dict[str, str]) -> str:
    """Flatten a corpus entry to the single string that gets indexed/embedded."""
    title = doc.get("title", "").strip()
    text = doc.get("text", "").strip()
    return f"{title}\n\n{text}".strip() if title else text


if __name__ == "__main__":
    corpus, queries, qrels = load_split("test")
    n_judged = sum(len(v) for v in qrels.values())
    print(f"corpus:  {len(corpus):,} documents")
    print(f"queries: {len(queries):,} test queries with judgments")
    print(f"qrels:   {n_judged:,} judged (query, doc) pairs "
          f"-- {n_judged / len(queries):.2f} relevant docs per query")
