"""
Step 3 -- Macro clustering: UMAP + HDBSCAN across ALL spans.

Reads:   data/spans.jsonl
         data/embeddings.npy
Writes:  data/macro_clusters.json

Each cluster record:
    {
      "label":       null,          <- filled by label.py
      "description": null,
      "span_ids":    [...],
      "centroid":    [...],         <- mean embedding of members
      "size":        int,
      "is_noise":    bool           <- True for cluster_id == -1
    }

Usage:
    python -m src.cluster_macro
"""

import json
import sys
import numpy as np
from pathlib import Path

import umap
import hdbscan

from src.config import MACRO_UMAP, MACRO_HDBSCAN

# Reads the triage-filtered problematic spans (written by apply_triage.py)
SPAN_FILE = Path("data/problematic_spans.jsonl")
EMB_FILE  = Path("data/problematic_embeddings.npy")
OUT       = Path("data/macro_clusters.json")



def run() -> None:
    # ---- Load data ----------------------------------------------------------
    for path in (SPAN_FILE, EMB_FILE):
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Run the previous steps first.")

    spans = [json.loads(l) for l in SPAN_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    arr   = np.load(EMB_FILE)

    if len(spans) != len(arr):
        sys.exit(
            f"ERROR: spans ({len(spans)}) and embeddings ({len(arr)}) are misaligned. "
            "Re-run embed.py to regenerate both files together."
        )

    print(f"Loaded {len(spans)} spans, embeddings shape {arr.shape}")

    # ---- UMAP ---------------------------------------------------------------
    n_comp = min(MACRO_UMAP["n_components"], len(arr) - 2)
    umap_params = {**MACRO_UMAP, "n_components": n_comp}

    print(f"Running UMAP (n_components={n_comp}, n_neighbors={umap_params['n_neighbors']}) ...")
    reducer = umap.UMAP(**umap_params)
    reduced = reducer.fit_transform(arr)
    print(f"  -> reduced to {reduced.shape}")

    # ---- HDBSCAN ------------------------------------------------------------
    print(f"Running HDBSCAN (min_cluster_size={MACRO_HDBSCAN['min_cluster_size']}) ...")
    clusterer = hdbscan.HDBSCAN(**MACRO_HDBSCAN)
    labels    = clusterer.fit_predict(reduced)

    unique_labels = sorted(set(labels.tolist()))
    n_noise   = int((labels == -1).sum())
    n_clusters = len([l for l in unique_labels if l != -1])
    print(f"  -> {n_clusters} clusters, {n_noise} noise spans")

    # ---- Build output -------------------------------------------------------
    clusters: dict[str, dict] = {}
    for cid in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == cid]
        clusters[str(cid)] = {
            "label":       None,
            "description": None,
            "span_ids":    [spans[i]["span_id"] for i in idx],
            "centroid":    arr[idx].mean(axis=0).tolist(),
            "size":        len(idx),
            "is_noise":    bool(cid == -1),
        }

    OUT.write_text(json.dumps(clusters, indent=2), encoding="utf-8")
    print(f"Saved macro clusters -> {OUT}")
    print("\nCluster sizes:")
    for cid, c in sorted(clusters.items(), key=lambda x: -x[1]["size"]):
        tag = " [NOISE]" if c["is_noise"] else ""
        print(f"  cluster {cid:>4}{tag}: {c['size']} spans")


if __name__ == "__main__":
    run()
