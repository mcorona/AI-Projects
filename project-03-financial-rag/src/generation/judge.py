"""
LLM-as-judge scoring, and the checks that keep the judge honest.

An LLM judge is a measuring instrument, and an uncalibrated instrument is
worse than no instrument because it produces numbers people believe. This
module therefore ships three things, not one:

  1. Two pointwise judges -- groundedness (is the answer supported by the
     passages it was given?) and correctness (does it actually answer the
     question, measured against the gold passages from the qrels?).
  2. Self-consistency: every judgment can be run twice and the
     disagreement rate reported. A judge that contradicts itself on 20% of
     items cannot resolve a 3-point difference between systems.
  3. A cross-model check: re-judging a subsample with a different model.
     The generator and the primary judge are both Claude, so self-
     preference bias is a live risk; the honest response is to measure
     whether the ranking survives a different judge, not to assert it does.

Judging runs through the Batches API (50% of standard pricing) because
nothing about an offline evaluation is latency-sensitive.

Author: Manuel Corona
"""

import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.generation.rag import _strict_schema

JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "claude-opus-5")
CROSS_JUDGE_MODEL = os.getenv("RAG_CROSS_JUDGE_MODEL", "claude-haiku-4-5")
MAX_TOKENS = 1500

GROUNDEDNESS_SYSTEM = """You audit whether an answer is grounded in the passages \
it was given. You are not judging whether the answer is correct, useful, or \
well written -- only whether each factual claim in it is supported by the \
passages.

Verdicts:
- grounded: every factual claim traces to the passages.
- partially_grounded: the substance is supported, but at least one factual \
claim is not in the passages.
- ungrounded: the central claim is absent from the passages, or contradicts them.

General financial common sense that the passages neither state nor contradict \
(e.g. "interest compounds") is not an unsupported claim. A specific number, \
rate, rule, or recommendation that appears nowhere in the passages is."""

CORRECTNESS_SYSTEM = """You grade a candidate answer to a personal-finance \
question against reference passages that human annotators marked as relevant \
to that question.

The reference passages are the ground truth for what a good answer covers. \
They are forum answers, so they may be informal or partial -- judge the \
substance, not the style, and do not penalise a candidate for being better \
written or more concise than the references.

Verdicts:
- correct: addresses the question and is consistent with the references.
- partially_correct: addresses the question but misses the main point of the \
references, or mixes a correct point with a wrong one.
- incorrect: contradicts the references, or answers a different question.
- no_answer: declines to answer or says the information is unavailable.

Grade "no_answer" strictly on the answer text, not on whether declining was \
reasonable -- whether abstention was the right call is measured separately."""


class GroundednessVerdict(BaseModel):
    verdict: Literal["grounded", "partially_grounded", "ungrounded"]
    unsupported_claims: List[str] = Field(
        default_factory=list, description="Claims in the answer not supported by the passages.")
    reasoning: str = Field(description="One or two sentences.")


class CorrectnessVerdict(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect", "no_answer"]
    reasoning: str = Field(description="One or two sentences.")


def groundedness_params(question: str, context: str, answer: str,
                        model: str = JUDGE_MODEL) -> dict:
    user = (f"Passages given to the answerer:\n\n{context}\n\n---\n\n"
            f"Question: {question}\n\nAnswer under audit:\n{answer}")
    return _params(GROUNDEDNESS_SYSTEM, user, GroundednessVerdict, model)


def correctness_params(question: str, reference: str, answer: str,
                       model: str = JUDGE_MODEL) -> dict:
    user = (f"Question: {question}\n\nReference passages (human-judged relevant):\n\n"
            f"{reference}\n\n---\n\nCandidate answer:\n{answer}")
    return _params(CORRECTNESS_SYSTEM, user, CorrectnessVerdict, model)


def _params(system: str, user: str, schema_cls: type[BaseModel], model: str) -> dict:
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"format": {"type": "json_schema", "schema": _strict_schema(schema_cls)}},
    }


# --- Checks that need no judge at all ------------------------------------

def citation_precision(citations: List[int], retrieved_ids: List[str],
                       relevant: dict) -> Optional[float]:
    """
    Fraction of the passages an answer cited that are actually relevant to
    the question per the human qrels.

    This is the one grounding signal in the whole pipeline that involves no
    model judgment whatsoever -- the answer names passage numbers, the
    qrels say which documents are relevant, and the two either agree or
    they don't. Returns None when nothing was cited.
    """
    if not citations:
        return None
    hits, total = 0, 0
    for c in citations:
        if 1 <= c <= len(retrieved_ids):
            total += 1
            if retrieved_ids[c - 1] in relevant:
                hits += 1
    return hits / total if total else 0.0


def invalid_citation_rate(citations: List[int], n_passages: int) -> float:
    """
    Fraction of cited passage numbers that don't exist in the context --
    a citation-shaped hallucination. Should be 0.
    """
    if not citations:
        return 0.0
    bad = sum(1 for c in citations if not (1 <= c <= n_passages))
    return bad / len(citations)
