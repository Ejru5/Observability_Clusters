# Hierarchical Failure Clustering for Mistral Traces — Implementation Guide

## 1. Problem Statement

Mistral observability data (spans/traces from LangGraph + Mistral Workflow SDK pipelines) contains problematic spans — latency issues, guardrail triggers, hallucinations, tool failures — that need to be:

1. **Categorized** into known buckets you already recognize (latency, guardrail, etc.)
2. **Sub-categorized** within each bucket (e.g. guardrail → sexual / PII / hate / etc.)
3. **Discovered** — surfaced automatically when a new failure mode shows up that doesn't match anything known yet

The core design principle: **separate what you already know from what you don't.** Known categories get cheap deterministic rules. Unknown categories get unsupervised clustering. A feedback loop lets discovered clusters graduate into rules over time, so the expensive ML path only ever works on what's genuinely new.

---

## 2. Architecture Overview

```
Span ingested
     │
     ▼
[Problem Scoring] ── not problematic ──► discard / log normally
     │
     │ problematic
     ▼
[Known-Category Rules] ── matched ──► tagged with category
     │                                      │
     │ unmatched                            ▼
     ▼                              ┌───────────────────┐
[Unclassified Queue]                │ Guardrail branch:  │
     │                              │ pull subcategory   │
     ▼ (batch, e.g. weekly)         │ direct from        │
[Embed → UMAP → HDBSCAN]            │ moderation API     │
     │                              │ (no clustering)    │
     ▼                              └───────────────────┘
[Auto-label clusters via LLM]                │
     │                                       ▼
     ▼                              ┌───────────────────┐
[Store cluster + centroid           │ Other branches:    │
 in Qdrant]                         │ re-embed scoped to │
     │                              │ parent category →  │
     ▼                              │ UMAP → HDBSCAN      │
[Recurring cluster?]                │ (fresh clustering)  │
     │ yes                          └───────────────────┘
     ▼
[Promote to rule in step 3]
```

Two clustering passes happen, and they're **calculated separately, not hierarchically in one embedding space**:

- **Macro pass**: distinguishes categories from each other (latency vs. guardrail vs. hallucination vs. unknown)
- **Micro pass**: runs independently *within* a parent category, on a feature representation tuned to what actually differentiates subtypes of that specific failure

Mixing both into one global embedding compresses two different signals into one representation and does neither well.

---

## 3. Phase 0 — Problem Scoring

Don't cluster everything. Only cluster spans that are actually problematic. Compute a composite score/flag per span:

| Signal | Source | Threshold idea |
|---|---|---|
| Latency z-score | per-stage rolling baseline (per endpoint/model) | z > 2–3 |
| RAGAS faithfulness / answer relevancy | your existing RAGAS eval pipeline | below configured floor |
| Guardrail / moderation flag | Mistral moderation API | any category flagged True |
| Exception / error status | span status field | non-2xx / exception raised |
| Retry count | span metadata | > 0, or > N |

```python
def compute_problem_score(span):
    flags = {
        "latency": span["latency_zscore"] > 2.5,
        "quality": span.get("ragas_faithfulness", 1.0) < 0.6,
        "guardrail": span.get("moderation_flagged", False),
        "error": span["status"] != "success",
        "retries": span.get("retry_count", 0) > 0,
    }
    return {
        "is_problematic": any(flags.values()),
        "flags": flags,
    }
```

Run this as a post-processing job right after ingestion (or as an MLflow custom metric), so every downstream step only ever touches the pre-filtered problematic subset instead of full trace volume.

---

## 4. Phase 1 — Guardrail Subcategory Extraction (no clustering needed)

Mistral's moderation API already classifies text across 9 policy categories with per-category scores, multi-label:

- Sexual
- Hate and Discrimination
- Violence and Threats
- Dangerous and Criminal Content
- Self-harm
- Health
- Financial
- Law
- PII (Personally Identifiable Information)

If a span is flagged by moderation, just store what it returned — don't rebuild this with embeddings, you'd only be reconstructing what the classifier already told you, with more noise:

```python
response = client.classifiers.moderate_chat(
    model="mistral-moderation-latest",
    inputs=[{"role": "user", "content": span_text}],
)

result = response.results[0]
span["guardrail_subcategories"] = [
    cat for cat, flagged in result.categories.items() if flagged
]
span["guardrail_scores"] = result.category_scores  # for severity ranking
```

A single span can trigger multiple categories — keep it multi-label, don't force one winner.

---

## 5. Phase 2 — Known-Category Rule Layer

Route every problematic span through explicit, deterministic rules first:

```python
def classify_known(span):
    if span["flags"]["guardrail"]:
        return "guardrail"  # subcategories already attached from Phase 1
    if span["flags"]["latency"]:
        return "latency"
    if span["flags"]["quality"]:
        return "hallucination"
    if span["flags"]["error"]:
        # tie to your validator stages where relevant
        return span.get("validator_stage_failed") or "tool_error"
    return "unclassified"
```

Tie the `tool_error` branch to the validator stages already in your claims pipeline (Claim Integrity, Medical, Treatment, Disability, Litigation, Fraud/Anomaly) so errors inherit that context for free.

Everything that falls through to `"unclassified"` is what the discovery pipeline below exists for.

---

## 6. Phase 3 — Discovery Clustering for Unclassified Spans

Run this as a batch job (start weekly, tune cadence based on volume).

**Step 1 — Build a text representation per span**

```python
def build_cluster_text(span):
    parts = [
        span.get("error_message", ""),
        span.get("exception_type", ""),
        span.get("input_summary", "")[:200],
        span.get("output_summary", "")[:200],
        f"model={span.get('model')} tool={span.get('tool_called')}",
    ]
    return " | ".join(p for p in parts if p)
```

**Step 2 — Embed with Mistral's embedding endpoint**

```python
embeddings = mistral_client.embeddings.create(
    model="mistral-embed",
    inputs=[build_cluster_text(s) for s in unclassified_spans],
)
vectors = [e.embedding for e in embeddings.data]
```

**Step 3 — Reduce dimensions, then cluster**

```python
import umap
import hdbscan
import numpy as np

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.0,
    n_components=10,
    metric="cosine",
)
reduced = reducer.fit_transform(np.array(vectors))

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=8,
    metric="euclidean",
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(reduced)  # -1 = noise / unclustered
```

**Step 4 — Rank clusters by growth, not raw size**

This mirrors Arize Phoenix's approach (their open-source clustering ranks by drift ratio between production and baseline, not just cluster population):

```python
def rank_by_growth(cluster_id, this_week_counts, last_week_counts):
    this_week = this_week_counts.get(cluster_id, 0)
    last_week = last_week_counts.get(cluster_id, 1)  # avoid div by zero
    return this_week / last_week
```

Surface clusters with the highest growth ratio first — a small but rapidly growing cluster is more urgent than a large, stable one you already know about.

---

## 7. Phase 4 — Auto-Label Discovered Clusters

Sample 5–10 members per cluster and ask a Mistral model to name it:

```python
prompt = f"""You're looking at {n} examples of failed spans from a claims-processing pipeline
that were grouped together by similarity. Examples:

{chr(10).join(f"- {ex}" for ex in sampled_examples)}

Give a short category name (3-5 words) and a one-sentence description of what's going wrong.
Respond as JSON: {{"name": "...", "description": "..."}}
"""

response = mistral_client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)
```

Persist the cluster's centroid (mean of its vectors) plus the generated label in Qdrant — this is what lets future spans get matched against it directly instead of re-clustering from scratch (see Phase 6).

---

## 8. Phase 5 — Subclustering Non-Guardrail Parent Categories

For categories that don't have a built-in classifier (latency, hallucination, tool_error), repeat the embed → cluster loop, but:

- **Scope it** to only spans already tagged with that parent category
- **Re-embed on a category-specific representation** — don't reuse the Phase 3 global embedding

| Parent category | What to embed instead |
|---|---|
| Latency | per-stage duration breakdown, which component/tool was in the critical path |
| Hallucination | the specific claim vs. retrieved-context mismatch, RAGAS sub-scores |
| Tool_error | exception type + tool signature + failed validator stage |

```python
def build_latency_subcluster_text(span):
    stage_times = span.get("stage_durations", {})
    slowest_stage = max(stage_times, key=stage_times.get, default="unknown")
    return f"slowest_stage={slowest_stage} durations={stage_times} tool={span.get('tool_called')}"
```

Run the same UMAP + HDBSCAN + auto-label steps as Phase 3–4, independently, with its own `min_cluster_size` tuned to that category's volume (guardrail and latency won't behave the same).

---

## 9. Phase 6 — Closing the Loop

Two things need to happen continuously:

**a) Match new spans against existing clusters before re-clustering**

```python
def match_to_existing_cluster(span_vector, qdrant_client, collection, threshold=0.85):
    hits = qdrant_client.search(
        collection_name=collection,
        query_vector=span_vector,
        limit=1,
    )
    if hits and hits[0].score >= threshold:
        return hits[0].payload["cluster_label"]
    return None  # falls back to the unclassified queue for the next batch run
```

**b) Promote recurring clusters into Phase 2's rule set**

Track how many consecutive batch runs a cluster keeps reappearing. Once a cluster is stable (e.g. 3+ consecutive weekly runs, consistent size or growth), hardcode a rule for it in Phase 2 — this shrinks what Phase 3 has to rediscover each time and keeps the expensive clustering path reserved for genuinely novel failures.

---

## 10. Storage Schema

Suggested span record shape:

```json
{
  "span_id": "abc123",
  "is_problematic": true,
  "parent_category": "latency",
  "subcategory": {
    "type": "cluster",
    "cluster_id": "lat_cluster_7",
    "label": "Retrieval-stage timeout on large document sets",
    "confidence": 0.91
  },
  "guardrail_subcategories": [],
  "ragas_scores": { "faithfulness": 0.42 },
  "stage_durations": { "retrieval": 4.2, "generation": 0.8 },
  "first_seen": "2026-08-01",
  "last_seen": "2026-09-10"
}
```

For guardrail spans, `subcategory` becomes a list (multi-label) sourced from Phase 1 instead of a single `cluster_id`.

Qdrant collection layout: one collection per parent category (or a single collection with a `parent_category` payload filter), storing cluster centroids + labels for the nearest-neighbor matching in Phase 6.

---

## 11. MistralScope Dashboard Integration

Surface this as a drill-down view:

- **Top level**: category volume over time (latency / guardrail / hallucination / tool_error / unclassified)
- **Drill into a category**: subcategory breakdown (guardrail's 9 policy types, or discovered subclusters for latency/hallucination)
- **Drill into a subcluster**: auto-generated label + description, sample spans, growth trend

This plugs into the three-zone pipeline architecture already underpinning MistralScope.

---

## 12. Reference — How Existing Platforms Approach This

| Platform | Mechanism |
|---|---|
| **Arize Phoenix** (open source) | UMAP → HDBSCAN on embeddings, clusters ranked by drift ratio (production vs. baseline imbalance) |
| **Galileo Signals** | Not classic clustering — LLM-based memory system; batches production traces, builds condensed "institutional knowledge," matches new traces against accumulated findings |
| **Braintrust Topics** | LLM labels every trace first (task/sentiment/issue), then clusters on top of those labels |
| **Datadog Patterns** | Described as semantic/topic clustering over traffic; exact algorithm not published |

This guide's design follows Phoenix's approach most closely (proven, open source, and matches what's feasible to build directly on Mistral's stack) while adding the guardrail-subcategory shortcut and the two-tier macro/micro split, which none of these platforms expose as a single coherent workflow.

---

## 13. Suggested Build Order

1. Phase 0 (problem scoring) + Phase 1 (guardrail subcategory) — mostly schema/plumbing, fast to ship
2. Phase 2 (known-category rules) — deterministic, no ML
3. Phase 3 + 4 (discovery clustering + auto-labeling) — the core ML lift, start here once 1–2 are stable
4. Phase 5 (subclustering) — extend once Phase 3 is producing clean parent-level clusters
5. Phase 6 (feedback loop) — add once you have a few weeks of clustering history to promote from
6. Phase 7/11 (dashboard) — wire in incrementally as each phase produces data worth displaying

## 14. Tech Stack Summary

- **Embeddings**: Mistral embedding endpoint (`mistral-embed`)
- **Moderation**: Mistral moderation API (`mistral-moderation-latest`)
- **Labeling**: Mistral chat completion (`mistral-large-latest`)
- **Dimensionality reduction**: `umap-learn`
- **Clustering**: `hdbscan`
- **Vector storage**: Qdrant (centroids + nearest-neighbor matching)
- **Trace source**: MLflow-instrumented LangGraph / Mistral Workflow SDK pipelines
- **Eval signal**: RAGAS (faithfulness, answer relevancy)
