"""
Step 2 -- Normalise raw spans and embed them with mistral-embed.

Reads:   data/raw_spans.jsonl
Writes:  data/spans.jsonl       (normalised, one span per line)
         data/embeddings.npy    (float32 array [N, embed_dim], row-aligned)

Usage:
    python -m src.embed
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

from mistralai.client import Mistral

from src.config import MISTRAL_API_KEY, EMBED_MODEL, MISTRAL_CLIENT_KWARGS

RAW_FILE  = Path("data/raw_spans.jsonl")
SPAN_FILE = Path("data/spans.jsonl")
EMB_FILE  = Path("data/embeddings.npy")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _parse_ms(raw: dict) -> float | None:
    """Compute duration in milliseconds from start/end ISO timestamps."""
    s, e = raw.get("start_time"), raw.get("end_time")
    if not (s and e):
        return None
    try:
        return (
            datetime.fromisoformat(str(e)) - datetime.fromisoformat(str(s))
        ).total_seconds() * 1000
    except Exception:
        return None


def normalize(raw: dict) -> dict:
    """
    Map a raw Mistral span dict -> our internal schema.

    The SDK v2 response exposes all fields at the top level as flat keys.
    span_attributes is a nested dict with gen_ai.* / http.* keys.
    """
    # span_attributes is a nested dict (may be None or empty)
    attrs: dict = raw.get("span_attributes") or {}

    span_id = raw.get("span_id") or ""

    # Input / output text -- SDK v2 uses top-level input_messages / output_messages
    input_text  = (raw.get("input_messages")
                   or attrs.get("gen_ai.input.messages")
                   or "")
    output_text = (raw.get("output_messages")
                   or attrs.get("gen_ai.output.messages")
                   or "")

    # Model: prefer response_model, then request_model
    model = (raw.get("response_model")
             or raw.get("request_model")
             or attrs.get("gen_ai.response.model")
             or attrs.get("gen_ai.request.model")
             or "")

    # Status: status_code field (e.g. "Ok", "Error", "Unset")
    status = (raw.get("status_code") or "Unset").lower()

    # Duration: duration_ns is nanoseconds; fall back to start/end timestamps
    dur_ns = raw.get("duration_ns")
    if dur_ns:
        duration_ms: float | None = dur_ns / 1_000_000
    else:
        duration_ms = _parse_ms(raw)

    return {
        "span_id":        span_id,
        "trace_id":       raw.get("trace_id", ""),
        "operation_name": (raw.get("operation_name")
                           or raw.get("span_name")
                           or ""),
        "model":          model,
        "tool_called":    (raw.get("tool_name")
                           or attrs.get("gen_ai.tool.name")
                           or ""),
        "duration_ms":    duration_ms,
        "status":         status,
        "error_message":  (raw.get("status_message")
                           or raw.get("error_type")
                           or attrs.get("exception.message")
                           or ""),
        "exception_type": (raw.get("error_type")
                           or attrs.get("exception.type")
                           or ""),
        "input_summary":  str(input_text)[:200],
        "output_summary": str(output_text)[:200],
        "input_tokens":   (raw.get("usage_input_tokens")
                           or attrs.get("gen_ai.usage.input_tokens")),
        "output_tokens":  (raw.get("usage_output_tokens")
                           or attrs.get("gen_ai.usage.output_tokens")),
        "timestamp":      raw.get("start_time", ""),
    }


# ---------------------------------------------------------------------------
# Cluster text builders
# ---------------------------------------------------------------------------

def build_macro_text(span: dict) -> str:
    """
    Broad text used for MACRO clustering.
    Goal: let HDBSCAN separate top-level failure modes (latency vs error vs
    quality regression etc.) purely from the content of the span.
    """
    parts = [
        span["error_message"],
        span["exception_type"],
        span["input_summary"],
        span["output_summary"],
        f"op={span['operation_name']}",
        f"model={span['model']}",
        f"tool={span['tool_called']}",
        f"status={span['status']}",
    ]
    return " | ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_batch(client: Mistral, texts: list[str]) -> list[list[float]]:
    """Embed a single batch (<= 512 items) and return list of vectors."""
    resp = client.embeddings.create(model=EMBED_MODEL, inputs=texts)
    return [e.embedding for e in resp.data]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed all texts in chunks of 64, return float32 array [N, dim]."""
    if not MISTRAL_API_KEY:
        sys.exit(
            "ERROR: MISTRAL_API_KEY is not set.\n"
            "Copy .env.example -> .env and add your key."
        )

    batch_size = 64
    vectors: list[list[float]] = []
    with Mistral(api_key=MISTRAL_API_KEY, **MISTRAL_CLIENT_KWARGS) as client:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            batch_vecs = _embed_batch(client, chunk)
            vectors.extend(batch_vecs)
            print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)} spans")

    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not RAW_FILE.exists() or RAW_FILE.stat().st_size == 0:
        sys.exit(f"ERROR: {RAW_FILE} not found or empty. Run fetch_spans first.")

    raw_lines = [l for l in RAW_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not raw_lines:
        sys.exit(f"ERROR: {RAW_FILE} contains no data.")

    print(f"Normalising {len(raw_lines)} raw spans ...")
    raws  = [json.loads(l) for l in raw_lines]
    spans = [normalize(r) for r in raws]

    texts = [build_macro_text(s) for s in spans]
    print(f"Embedding {len(texts)} spans with {EMBED_MODEL} ...")
    arr = embed_texts(texts)

    # Save normalised spans
    with open(SPAN_FILE, "w", encoding="utf-8") as f:
        for s in spans:
            f.write(json.dumps(s) + "\n")
    print(f"Saved normalised spans -> {SPAN_FILE}")

    # Save embeddings
    np.save(EMB_FILE, arr)
    print(f"Saved embeddings {arr.shape} -> {EMB_FILE}")


if __name__ == "__main__":
    main()

run = main
