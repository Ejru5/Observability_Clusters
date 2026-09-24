"""
Step 2b -- Embed trace-level text summaries with mistral-embed.

Calls rollup_traces internally first (normalise + rollup), then embeds
the trace_text field for every trace.

Reads:   data/raw_spans.jsonl
Writes:  data/spans.jsonl             (normalised spans — intermediate)
         data/traces.jsonl            (one record per trace)
         data/trace_embeddings.npy    (float32 [N_traces, dim], row-aligned)

Usage:
    python -m src.embed
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path

from mistralai.client import Mistral

from src.config import MISTRAL_API_KEY, EMBED_MODEL, MISTRAL_CLIENT_KWARGS
from src.rollup_traces import main as rollup_main

TRACE_FILE = Path("data/traces.jsonl")
EMB_FILE   = Path("data/trace_embeddings.npy")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_batch(client: Mistral, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, inputs=texts)
    return [e.embedding for e in resp.data]


def embed_texts(texts: list[str]) -> np.ndarray:
    if not MISTRAL_API_KEY:
        sys.exit("ERROR: MISTRAL_API_KEY is not set.")

    batch_size = 64
    vectors: list[list[float]] = []

    with Mistral(api_key=MISTRAL_API_KEY, **MISTRAL_CLIENT_KWARGS) as client:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            batch_vecs = _embed_batch(client, chunk)
            vectors.extend(batch_vecs)
            print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)} traces")

    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1: normalise + rollup
    print("=== Step 1: Normalise spans and rollup to traces ===")
    rollup_main()

    # Step 2: load traces
    if not TRACE_FILE.exists() or TRACE_FILE.stat().st_size == 0:
        sys.exit(f"ERROR: {TRACE_FILE} empty after rollup.")

    traces = [
        json.loads(l)
        for l in TRACE_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    if not traces:
        sys.exit("ERROR: No traces to embed.")

    texts = [t["trace_text"] for t in traces]

    print(f"\n=== Step 2: Embedding {len(texts)} traces with {EMBED_MODEL} ===")
    arr = embed_texts(texts)

    np.save(EMB_FILE, arr)
    print(f"Saved embeddings {arr.shape} -> {EMB_FILE}")


if __name__ == "__main__":
    main()

run = main
