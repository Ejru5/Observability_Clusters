"""
Step 5 -- Cluster error traces by facet similarity, then LLM-label each cluster.

No UMAP, no HDBSCAN.

Algorithm:
  1. Compute pairwise cosine similarity on facet_embeddings.
  2. Agglomerative clustering (average linkage, cosine distance threshold).
  3. Traces that end up alone (singletons) are tagged as "unique_error".
  4. LLM labels each non-singleton cluster with a name + description.

Reads:   data/trace_facets.jsonl
         data/facet_embeddings.npy
         data/error_traces.jsonl
Writes:  data/trace_clusters.json

Output schema:
{
  "clusters": {
    "0": {
      "cluster_id":             "0",
      "label":                  "Tool Connection Timeout",
      "description":            "...",
      "is_unique_error":        false,
      "trace_ids":              [...],
      "size":                   12,
      "dominant_issue_type":    "Tool Connection Timeout",
      "affected_components":    ["tool"],
      "representative_trace_id": "abc123",
    },
    ...
  },
  "unique_errors": [
    {
      "trace_id":     "xyz",
      "issue_type":   "...",
      "summary":      "...",
      "component":    "...",
    },
    ...
  ]
}

Usage:
    python -m src.cluster_traces
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter

from src.config import (
    MISTRAL_API_KEY,
    LABEL_MODEL,
    CLUSTER_DISTANCE_THRESHOLD,
    CLUSTER_MIN_SIZE,
    MISTRAL_CLIENT_KWARGS,
)

FACETS_FILE    = Path("data/trace_facets.jsonl")
FACET_EMB_FILE = Path("data/facet_embeddings.npy")
ERROR_FILE     = Path("data/error_traces.jsonl")
OUT_FILE       = Path("data/trace_clusters.json")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_by_cosine(
    embeddings: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    """
    Agglomerative clustering with average linkage on cosine distance.
    Returns integer label array [N] — each unique int is a cluster ID.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    if len(embeddings) == 1:
        return np.array([0])

    normed = normalize(embeddings, norm="l2")

    # Agglomerative clustering with cosine metric
    # For small N we can use precomputed distance matrix; for large N use the
    # connectivity-free variant which sklearn handles via cosine metric directly.
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(normed)
    except Exception as exc:
        print(f"  WARNING: AgglomerativeClustering failed ({exc}). Assigning all to one cluster.")
        labels = np.zeros(len(embeddings), dtype=int)

    return labels.astype(int)


# ---------------------------------------------------------------------------
# LLM labelling
# ---------------------------------------------------------------------------

def _heuristic_label(facets: list[dict]) -> dict:
    issues = [f.get("issue_type", "") for f in facets]
    comps  = [f.get("affected_component", "unknown") for f in facets]
    top_issue = Counter(issues).most_common(1)[0][0] if issues else "Unknown Error"
    top_comp  = Counter(comps).most_common(1)[0][0] if comps else "unknown"
    return {
        "name":        top_issue,
        "description": f"Cluster of {len(facets)} traces with '{top_issue}' failure pattern.",
    }


def _llm_label(client, facets: list[dict], cluster_size: int) -> dict:
    from mistralai.client import Mistral

    samples = facets[:5]
    example_block = "\n".join(
        f"- issue_type: {f.get('issue_type')} | component: {f.get('affected_component')} | summary: {f.get('summary')}"
        for f in samples
    )

    prompt = (
        f"You are an expert LLM observability engineer.\n"
        f"These {cluster_size} error traces share a similar failure pattern:\n\n"
        f"{example_block}\n\n"
        "Generate a cluster name (3-5 words) and one-sentence description of the shared failure.\n"
        "Respond as JSON only:\n"
        '{"name": "...", "description": "..."}'
    )

    try:
        resp = client.chat.complete(
            model=LABEL_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content.strip())
        if "name" in data and "description" in data:
            return data
    except Exception as exc:
        print(f"    WARNING: LLM label failed ({exc}), using heuristic.")

    return _heuristic_label(facets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    for path in (FACETS_FILE, FACET_EMB_FILE, ERROR_FILE):
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Run facet_extract step first.")

    facets: list[dict] = [
        json.loads(l)
        for l in FACETS_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    embeddings: np.ndarray = np.load(FACET_EMB_FILE)
    error_traces: list[dict] = [
        json.loads(l)
        for l in ERROR_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    if len(facets) != len(embeddings):
        sys.exit(f"ERROR: facets ({len(facets)}) and embeddings ({len(embeddings)}) misaligned.")

    n = len(facets)
    print(f"Clustering {n} error traces by facet similarity ...")
    print(f"  Distance threshold: {CLUSTER_DISTANCE_THRESHOLD} (similarity > {1-CLUSTER_DISTANCE_THRESHOLD:.2f})")

    if n == 0:
        OUT_FILE.write_text(json.dumps({"clusters": {}, "unique_errors": []}, indent=2))
        print("No error traces — empty cluster file written.")
        return

    # Build a trace_id -> error_trace lookup
    trace_by_id: dict[str, dict] = {t["trace_id"]: t for t in error_traces}
    # Build a trace_id -> facet lookup
    facet_by_id: dict[str, dict] = {f["trace_id"]: f for f in facets}

    # --- Cluster ---
    labels = cluster_by_cosine(embeddings, CLUSTER_DISTANCE_THRESHOLD)
    unique_labels = sorted(set(labels.tolist()))
    print(f"  -> {len(unique_labels)} raw clusters found")

    # --- Group facets by cluster label ---
    groups: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(int(lbl), []).append(i)

    # --- Split singletons vs multi-trace clusters ---
    singleton_indices: list[int] = []
    multi_groups: dict[int, list[int]] = {}

    for lbl, members in groups.items():
        if len(members) < max(CLUSTER_MIN_SIZE + 1, 2):
            singleton_indices.extend(members)
        else:
            multi_groups[lbl] = members

    print(f"  -> {len(multi_groups)} clusters, {len(singleton_indices)} unique errors (singletons)")

    # --- LLM label multi-trace clusters ---
    client = None
    if MISTRAL_API_KEY:
        try:
            from mistralai.client import Mistral
            client = Mistral(api_key=MISTRAL_API_KEY, **MISTRAL_CLIENT_KWARGS)
        except Exception as e:
            print(f"  WARNING: Mistral client init failed ({e}). Using heuristic labels.")

    clusters_out: dict[str, dict] = {}

    for cluster_idx, (lbl, members) in enumerate(
        sorted(multi_groups.items(), key=lambda x: -len(x[1]))
    ):
        member_facets = [facets[i] for i in members]
        cluster_id    = str(cluster_idx)

        print(f"  Labelling cluster {cluster_id} ({len(members)} traces) ...")
        if client:
            label_data = _llm_label(client, member_facets, len(members))
        else:
            label_data = _heuristic_label(member_facets)

        # Dominant issue type and components
        issue_types = [f.get("issue_type", "") for f in member_facets]
        components  = [f.get("affected_component", "unknown") for f in member_facets]
        dom_issue   = Counter(issue_types).most_common(1)[0][0] if issue_types else ""
        dom_comps   = [c for c, _ in Counter(components).most_common()]

        trace_ids = [facets[i]["trace_id"] for i in members]
        rep_id    = trace_ids[0]  # representative = first (facets sorted by similarity)

        clusters_out[cluster_id] = {
            "cluster_id":              cluster_id,
            "label":                   label_data.get("name", dom_issue),
            "description":             label_data.get("description", ""),
            "is_unique_error":         False,
            "trace_ids":               trace_ids,
            "size":                    len(members),
            "dominant_issue_type":     dom_issue,
            "affected_components":     dom_comps,
            "representative_trace_id": rep_id,
        }
        print(f"    -> \"{label_data.get('name')}\"")

    if client:
        try:
            client.__exit__(None, None, None)
        except Exception:
            pass

    # --- Unique errors (singletons) ---
    unique_errors_out: list[dict] = []
    for i in singleton_indices:
        f = facets[i]
        unique_errors_out.append({
            "trace_id":  f["trace_id"],
            "issue_type": f.get("issue_type", ""),
            "summary":    f.get("summary", ""),
            "component":  f.get("affected_component", "unknown"),
            "root_cause": f.get("root_cause", ""),
        })

    result = {
        "clusters":      clusters_out,
        "unique_errors": unique_errors_out,
    }
    OUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {len(clusters_out)} clusters + {len(unique_errors_out)} unique errors -> {OUT_FILE}")


if __name__ == "__main__":
    run()

main = run
