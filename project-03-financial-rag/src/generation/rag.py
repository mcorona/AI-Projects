"""
Answer generation: the "G" in RAG, plus the two conditions it is measured
against.

Three conditions are generated for every evaluation question, because
"the RAG answer looked good" is not a result:

  closed_book  no context at all -- what the model already knows about
               personal finance. If RAG does not beat this, retrieval is
               contributing nothing.
  rag          top-k documents from the retriever under test.
  oracle       the gold relevant documents from the qrels. This is the
               ceiling: it separates *retrieval* failure from *generation*
               failure. When oracle is strong and rag is weak, the fix is
               a better retriever; when oracle is also weak, the fix is the
               prompt, the model, or the benchmark.

Answers are structured (answer / citations / context_sufficient) so the
citation check is mechanical rather than a regex over prose.

Author: Manuel Corona
"""

import os
from typing import Dict, List, Optional, Sequence

import anthropic
from pydantic import BaseModel, Field

# Opus 5 for generation. The judge (src/generation/judge.py) runs on the
# same family, which is a self-preference risk documented in MODEL_CARD.md
# and probed with a cross-model check rather than waved away.
GENERATOR_MODEL = os.getenv("RAG_GENERATOR_MODEL", "claude-opus-5")
MAX_TOKENS = 1500

SYSTEM_PROMPT = """You answer personal-finance and investing questions for a \
retrieval-augmented QA system that is being evaluated for factual grounding.

Rules:
- Answer in at most 120 words, in plain prose. No preamble, no headings.
- When context passages are provided, ground every factual claim in them and \
cite the passage numbers you used, e.g. [2]. Do not cite a passage you did not use.
- If the provided passages do not contain enough information to answer, set \
context_sufficient to false and say plainly what is missing. Do not fall back \
on your own knowledge to paper over a retrieval failure -- an honest "the \
provided sources don't cover this" is the correct answer in that case.
- When no context is provided at all, answer from your own knowledge and \
leave citations empty."""


class RagAnswer(BaseModel):
    """Structured answer shape shared by all three conditions."""
    answer: str = Field(description="The answer, at most 120 words.")
    citations: List[int] = Field(
        default_factory=list,
        description="1-based passage numbers actually used. Empty when no context was given.",
    )
    context_sufficient: bool = Field(
        default=True,
        description="False when the provided passages do not support an answer.",
    )


def format_context(doc_ids: Sequence[str], corpus: Dict[str, Dict[str, str]],
                   max_chars: int = 1200) -> str:
    """
    Render retrieved documents as numbered passages.

    Passages are truncated at max_chars. FiQA documents are forum answers
    and the long tail runs to several thousand characters; without a cap,
    a handful of queries would dominate the token bill and the model would
    be reading past the part the retriever actually matched on.
    """
    from src.data.loader import doc_text
    blocks = []
    for i, did in enumerate(doc_ids, start=1):
        text = doc_text(corpus[did]).replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"[{i}] {text}")
    return "\n\n".join(blocks)


def build_messages(question: str, context: Optional[str]) -> List[dict]:
    if context is None:
        user = f"Question: {question}"
    else:
        user = f"Context passages:\n\n{context}\n\n---\n\nQuestion: {question}"
    return [{"role": "user", "content": user}]


def answer(client: anthropic.Anthropic, question: str,
           context: Optional[str] = None, model: str = GENERATOR_MODEL) -> RagAnswer:
    """Generate one answer synchronously. Used by the app and by smoke tests."""
    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=build_messages(question, context),
        output_format=RagAnswer,
    )
    return response.parsed_output


def batch_params(question: str, context: Optional[str],
                 model: str = GENERATOR_MODEL) -> dict:
    """
    The same request as `answer()`, shaped for the Batches API.

    The evaluation generates 3 conditions x N questions in one go and none
    of it is latency-sensitive, so it runs through batches at 50% of
    standard pricing. `output_config.format` is spelled out here because
    the batch path has no `messages.parse` equivalent -- the schema comes
    from the same Pydantic model, so the two paths cannot drift.
    """
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": build_messages(question, context),
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": _strict_schema(RagAnswer),
            }
        },
    }


def _strict_schema(model_cls: type[BaseModel]) -> dict:
    """
    Pydantic's JSON schema, tightened for the structured-output contract:
    every property required and no additional properties, recursively.
    """
    schema = model_cls.model_json_schema()

    def tighten(node: dict):
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"])
            node["additionalProperties"] = False
            for child in node["properties"].values():
                tighten(child)
        if "items" in node and isinstance(node["items"], dict):
            tighten(node["items"])

    tighten(schema)
    return schema
