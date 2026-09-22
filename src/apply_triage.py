"""
Apply triage: filter spans by the selected algorithm and write output files.

This is a lightweight step called by the API when the user toggles
between LOF and Autoencoder in the UI.  It reads the pre-computed
triage_scores.json (written by triage.py) and produces:
  - data/problematic_spans.jsonl       (input to cluster_macro.py)
  - data/problematic_embeddings.npy    (aligned embeddings)

It also invalidates cached downstream files (points_2d.json, macro_clusters.json)
so the pipeline re-runs from this point.

Reads:   data/triage_scores.json
         data/spans.jsonl
         data/embeddings.npy
Writes:  data/problematic_spans.jsonl
         data/problematic_embeddings.npy

Usage:
    python -m src.apply_triage --algo lof
    python -m src.apply_triage --algo ae
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path

SCORES_FILE   = Path("data/triage_scores.json")
SPAN_FILE     = Path("data/spans.jsonl")
EMB_FILE      = Path("data/embeddings.npy")

PROB_SPAN_FILE = Path("data/problematic_spans.jsonl")
PROB_EMB_FILE  = Path("data/problematic_embeddings.npy")

# Downstream caches to invalidate when algo changes
INVALIDATE = [
    Path("data/points_2d.json"),
]


def apply(algorithm: str) -> dict:
    """
    Apply the selected algorithm and write output files.
    Returns a summary dict.
    """
    algorithm = algorithm.lower().strip()
    if algorithm not in ("lof", "ae"):
        raise ValueError(f"Unknown algorithm '{algorithm}'. Must be 'lof' or 'ae'.")

    for path in (SCORES_FILE, SPAN_FILE, EMB_FILE):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run triage step first.")

    # Load all data
    scores: list[dict] = json.loads(SCORES_FILE.read_text(encoding="utf-8"))
    spans: list[dict] = [
        json.loads(l)
        for l in SPAN_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    embeddings: np.ndarray = np.load(EMB_FILE)

    if len(spans) != len(embeddings) or len(spans) != len(scores):
        raise ValueError(
            f"Misaligned data: spans={len(spans)}, "
            f"embeddings={len(embeddings)}, scores={len(scores)}. "
            "Re-run embed + triage steps."
        )

    # Build span_id -> score lookup
    score_by_id = {r["span_id"]: r for r in scores}

    # Select the is_problematic field for the active algorithm
    prob_key = f"is_problematic_{algorithm}"

    # Filter
    prob_indices = [
        i for i, s in enumerate(spans)
        if score_by_id.get(s["span_id"], {}).get(prob_key, False)
    ]

    if not prob_indices:
        print(f"WARNING: No problematic spans found for algorithm '{algorithm}'.")
        # Write empty files so downstream steps don't crash
        PROB_SPAN_FILE.write_text("", encoding="utf-8")
        np.save(PROB_EMB_FILE, np.empty((0, embeddings.shape[1]), dtype=np.float32))
    else:
        prob_spans      = [spans[i] for i in prob_indices]
        prob_embeddings = embeddings[prob_indices]

        # Attach triage metadata to each span
        algo_types_key = f"problem_types_{algorithm}"
        for s in prob_spans:
            rec = score_by_id.get(s["span_id"], {})
            s["is_problematic"]   = True
            s["problem_types"]    = rec.get(algo_types_key, [])
            s["lof_score"]        = rec.get("lof_score")
            s["ae_recon_error"]   = rec.get("ae_reconstruction_error")
            s["triage_algorithm"] = algorithm

        with open(PROB_SPAN_FILE, "w", encoding="utf-8") as f:
            for s in prob_spans:
                f.write(json.dumps(s) + "\n")

        np.save(PROB_EMB_FILE, prob_embeddings)

    # Invalidate downstream caches
    for path in INVALIDATE:
        if path.exists():
            try:
                path.unlink()
                print(f"  Invalidated cache: {path.name}")
            except Exception:
                pass

    n_prob    = len(prob_indices)
    n_healthy = len(spans) - n_prob
    print(f"Applied algorithm='{algorithm}': {n_prob} problematic, {n_healthy} healthy")
    print(f"  -> {PROB_SPAN_FILE}")
    print(f"  -> {PROB_EMB_FILE}")

    return {
        "algorithm":   algorithm,
        "total_spans": len(spans),
        "problematic": n_prob,
        "healthy":     n_healthy,
    }


def run() -> None:
    parser = argparse.ArgumentParser(description="Apply triage algorithm to filter problematic spans.")
    parser.add_argument(
        "--algo",
        choices=["lof", "ae"],
        default="lof",
        help="Algorithm to apply: lof or ae (default: lof)",
    )
    args = parser.parse_args()
    apply(args.algo)


if __name__ == "__main__":
    run()

main = run
