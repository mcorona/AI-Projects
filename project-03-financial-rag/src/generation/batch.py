"""
Thin wrapper over the Message Batches API for offline evaluation runs.

An evaluation is the textbook batch workload: a few thousand independent
requests, no user waiting on any of them, and a 50% discount for saying so.
Everything here is restartable -- batch ids are persisted, so a poll that
gets interrupted does not throw away work that has already been paid for.

Author: Manuel Corona
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = ROOT / "data" / "processed" / "batches"

# Claude API list prices per million tokens, halved for batch. Used only to
# report what a run cost -- the numbers printed by the eval are estimates,
# not an invoice.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
BATCH_DISCOUNT = 0.5


def submit(client: anthropic.Anthropic, requests: List[Tuple[str, dict]], tag: str) -> str:
    """Submit a batch and persist its id under data/processed/batches/<tag>.json."""
    batch = client.messages.batches.create(
        requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params))
                  for cid, params in requests]
    )
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    (BATCH_DIR / f"{tag}.json").write_text(json.dumps({
        "batch_id": batch.id, "tag": tag, "n_requests": len(requests),
        "submitted_at": time.time(),
    }, indent=2))
    print(f"[{tag}] submitted {len(requests)} requests as {batch.id}")
    return batch.id


def load_batch_id(tag: str):
    path = BATCH_DIR / f"{tag}.json"
    return json.loads(path.read_text())["batch_id"] if path.exists() else None


def wait(client: anthropic.Anthropic, batch_id: str, poll_seconds: int = 30) -> None:
    """Block until the batch ends. Most finish well inside an hour."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        if batch.processing_status == "ended":
            print(f"[{batch_id}] ended -- {counts.succeeded} ok, {counts.errored} errored")
            return
        print(f"[{batch_id}] {batch.processing_status}: "
              f"{counts.processing} processing, {counts.succeeded} done")
        time.sleep(poll_seconds)


def collect(client: anthropic.Anthropic, batch_id: str) -> Tuple[Dict[str, dict], Dict[str, int]]:
    """
    Fetch results keyed by custom_id.

    Results arrive in arbitrary order, so they are keyed by custom_id and
    never by position. Structured-output responses are parsed here; a
    request that errored is recorded as an error entry rather than dropped,
    so the analysis can report how many items it actually scored.
    """
    out: Dict[str, dict] = {}
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_read_input_tokens": 0, "n_errors": 0}
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type != "succeeded":
            out[cid] = {"error": result.result.type}
            usage["n_errors"] += 1
            continue
        msg = result.result.message
        usage["input_tokens"] += msg.usage.input_tokens
        usage["output_tokens"] += msg.usage.output_tokens
        usage["cache_read_input_tokens"] += getattr(msg.usage, "cache_read_input_tokens", 0) or 0
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            out[cid] = {"parsed": json.loads(text)}
        except json.JSONDecodeError:
            # output_config.format should make this impossible; recorded
            # rather than swallowed so it shows up in the error count.
            out[cid] = {"error": "unparseable", "raw": text[:500]}
            usage["n_errors"] += 1
    return out, usage


def estimate_cost(usage: Dict[str, int], model: str) -> float:
    """Approximate USD cost of a completed batch."""
    in_price, out_price = PRICES.get(model, (5.0, 25.0))
    return BATCH_DISCOUNT * (
        usage["input_tokens"] / 1e6 * in_price
        + usage["output_tokens"] / 1e6 * out_price
    )


def run(client: anthropic.Anthropic, requests: List[Tuple[str, dict]], tag: str,
        model: str, resume: bool = True):
    """Submit (or resume), wait, and collect one batch. Returns (results, usage)."""
    batch_id = load_batch_id(tag) if resume else None
    if batch_id:
        print(f"[{tag}] resuming {batch_id}")
    else:
        batch_id = submit(client, requests, tag)
    wait(client, batch_id)
    results, usage = collect(client, batch_id)
    usage["estimated_usd"] = round(estimate_cost(usage, model), 2)
    print(f"[{tag}] {usage['input_tokens']:,} in / {usage['output_tokens']:,} out "
          f"~= ${usage['estimated_usd']:.2f} (batch pricing)")
    return results, usage
