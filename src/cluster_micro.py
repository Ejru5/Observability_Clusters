"""
Step 4 -- Micro clustering: per-macro re-embed + UMAP + HDBSCAN.

For each macro cluster (excluding noise) we:
  1. Build a *different*, more detailed text representation of each span
     (focused on fine-grained differences within the cluster rather than
      separating clusters from each other).
  2. Re-embed those texts with mistral-embed.
  3. Run UMAP + HDBSCAN independently on the resulting vectors.

This is intentionally a separate embedding pass -- reusing the macro
embeddings would compress both signals into one space and do neither well.

Reads:   data/spans.jsonl
         data/macro_clusters.json
Writes:  data/micro_clusters.json

Structure:
    {
      "<macro_id>": {
        "<micro_id>": {
          "label":       null,
          "description": null,
          "span_ids":    [...],
          "centroid":    [...],
          "size":        int,
          "is_noise":    bool
        }
      }
    }

Usage:
    python -m src.cluster_micro
"""

import json
import sys
import numpy as np
from pathlib import Path

import umap
import hdbscan
from mistralai.client import Mistral

from src.config import MISTRAL_API_KEY, EMBED_MODEL, MICRO_UMAP, MICRO_HDBSCAN, MICRO_MIN_SPANS, MISTRAL_CLIENT_KWARGS

SPAN_FILE      = Path("data/spans.jsonl")
PROB_SPAN_FILE = Path("data/problematic_spans.jsonl")
MACRO_FILE     = Path("data/macro_clusters.json")
OUT            = Path("data/micro_clusters.json")


# ---------------------------------------------------------------------------
# Micro-level text representation
# ---------------------------------------------------------------------------

def build_micro_text(span: dict) -> str:
    """
    Scoped text for MICRO clustering.

    Unlike the macro text (which blends error + input + output to separate
    *categories*), the micro text emphasises fine-grained operational detail:
    exact error wording, duration, tool context, specific output phrasing.
    """
    duration = (
        f"duration={span['duration_ms']:.0f}ms"
        if span.get("duration_ms") is not None
        else ""
    )
    clean_types = [t for t in span.get("problem_types", []) if t not in ("ml_anomaly", "ml_confirmed")]
    problem_str = f"problem={','.join(clean_types)}" if clean_types else ""
    parts = [
        problem_str,
        duration,
        f"status={span['status']}",
        span.get("exception_type") or "",
        span.get("error_message") or "",
        span.get("output_summary") or "",
        span.get("input_summary") or "",
        f"op={span.get('operation_name', '')}",
        f"tool={span.get('tool_called', '')}",
        f"model={span.get('model', '')}",
    ]
    return " | ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(client: Mistral, texts: list[str]) -> np.ndarray:
    batch_size = 64
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp  = client.embeddings.create(model=EMBED_MODEL, inputs=chunk)
        vectors.extend([e.embedding for e in resp.data])
    return np.array(vectors, dtype=np.float32)


def _subcluster(arr: np.ndarray) -> np.ndarray:
    """UMAP -> HDBSCAN on a subset array. Returns label array."""
    n_comp = min(MICRO_UMAP["n_components"], len(arr) - 2)
    n_nbrs = min(MICRO_UMAP["n_neighbors"], len(arr) - 1)
    params = {**MICRO_UMAP, "n_components": n_comp, "n_neighbors": n_nbrs}

    reduced = umap.UMAP(**params).fit_transform(arr)
    labels  = hdbscan.HDBSCAN(**MICRO_HDBSCAN).fit_predict(reduced)
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    if not MACRO_FILE.exists():
        sys.exit(f"ERROR: {MACRO_FILE} not found. Run the previous steps first.")

    all_spans: dict[str, dict] = {}
    if SPAN_FILE.exists():
        for l in SPAN_FILE.read_text(encoding="utf-8").splitlines():
            if l.strip():
                s = json.loads(l)
                all_spans[s["span_id"]] = s
    if PROB_SPAN_FILE.exists():
        for l in PROB_SPAN_FILE.read_text(encoding="utf-8").splitlines():
            if l.strip():
                s = json.loads(l)
                all_spans[s["span_id"]] = s
    macro_data = json.loads(MACRO_FILE.read_text(encoding="utf-8"))

    micro_out: dict[str, dict] = {}

    if not MISTRAL_API_KEY:
        sys.exit("ERROR: MISTRAL_API_KEY is not set.")

    with Mistral(api_key=MISTRAL_API_KEY, **MISTRAL_CLIENT_KWARGS) as client:
        for macro_id, macro_cluster in macro_data.items():
            if macro_cluster["is_noise"]:
                micro_out[macro_id] = {}
                continue

            span_ids = macro_cluster["span_ids"]
            members  = [all_spans[sid] for sid in span_ids if sid in all_spans]

            if len(members) < MICRO_MIN_SPANS:
                print(f"  macro {macro_id}: only {len(members)} spans -- skipping subcluster")
                micro_out[macro_id] = {}
                continue

            print(f"  macro {macro_id} ({len(members)} spans): embedding ...", end=" ", flush=True)
            texts = [build_micro_text(s) for s in members]
            arr   = _embed(client, texts)

            print("clustering ...", end=" ", flush=True)
            labels = _subcluster(arr)

            sub: dict[str, dict] = {}
            for cid in sorted(set(labels.tolist())):
                idx = [i for i, l in enumerate(labels) if l == cid]
                sub[str(cid)] = {
                    "label":       None,
                    "description": None,
                    "span_ids":    [members[i]["span_id"] for i in idx],
                    "centroid":    arr[idx].mean(axis=0).tolist(),
                    "size":        len(idx),
                    "is_noise":    bool(cid == -1),
                }

            n_real  = sum(1 for c in sub.values() if not c["is_noise"])
            n_noise = sum(1 for c in sub.values() if c["is_noise"])
            noise_count = next((c["size"] for c in sub.values() if c["is_noise"]), 0)
            print(f"-> {n_real} subclusters, {noise_count} noise spans")
            micro_out[macro_id] = sub

    OUT.write_text(json.dumps(micro_out, indent=2), encoding="utf-8")
    print(f"\nSaved micro clusters -> {OUT}")


if __name__ == "__main__":
    run()
