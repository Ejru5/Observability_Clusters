# Hierarchical Failure Clustering on Mistral Observability API Data

Apply the 6-phase pipeline from [`mistral-trace-clustering-guide.md`](file:///c:/Users/Dhruvkumar/Desktop/Observability_Clusters/mistral-trace-clustering-guide.md) using **only** the Mistral Observability Beta API as the data source.

---

## Data Source — What the Mistral API Actually Returns

The Mistral Python SDK exposes a `Beta.Observability` namespace with two resources:

### `mistral.beta.observability.traces` — Trace-level operations
| Method | REST path | Purpose |
|---|---|---|
| `.search(page_size, from_, to_, ...)` | `POST /v1/observability/traces/search` | Paginate through all traces |
| `.get_trace_by_id(trace_id)` | `GET /v1/observability/traces/{id}` | Single trace record |
| `.get_trace_spans(trace_id)` | `GET /v1/observability/traces/{id}/spans` | All spans inside one trace |
| `.get_span_by_id(span_id)` | `GET /v1/observability/traces/.../spans/{id}` | Single span |
| `.aggregate(...)` | `POST /v1/observability/traces/aggregate` | Server-side aggregation |
| `.get_trace_fields()` | `GET /v1/observability/traces/fields` | Schema discovery |

### `mistral.beta.observability.spans` — Span-level operations (cross-trace)
| Method | REST path | Purpose |
|---|---|---|
| `.search_spans(page_size, from_, to_, ...)` | `POST /v1/observability/spans/search` | Search spans across all traces |
| `.aggregate(...)` | `POST /v1/observability/spans/aggregate` | Server-side aggregation |
| `.search_span_evaluations(...)` | `POST /v1/observability/spans/evaluations/search` | Fetch evaluation scores per span |
| `.list_span_fields()` | `GET /v1/observability/spans/fields` | Schema discovery |

> [!NOTE]
> Data retention: **30 days** rolling. Always bound fetches with `from_`/`to_` date params.

> [!IMPORTANT]
> Both APIs are under `mistral.beta.*` (preview). Always call `list_span_fields()` first on startup to confirm the current live field schema before mapping field names in `normalize.py`.

---

## Span Fields Available for Scoring

Based on the published observability schema (verify via `list_span_fields()`):

| Field | Used for |
|---|---|
| `span_id`, `trace_id` | Identity |
| `name` / `operation_type` | Stage (generation, retrieval, tool call, etc.) |
| `start_time`, `end_time` → `duration_ms` | Latency scoring |
| `status` (`ok` / `error`) | Error flag |
| `status_message` / `error_message` | Error text for clustering |
| `model` | Which Mistral model was called |
| `input_tokens`, `output_tokens` | Token usage |
| `attributes.input` / `attributes.output` | Request/response text (for embeddings) |
| `attributes.tool_name` | Tool called, if any |
| Span evaluations (via `search_span_evaluations`) | Quality scores (faithfulness, etc.) |

---

## One Remaining Open Question

> [!NOTE]
> **Qdrant vs SQLite for centroid storage** (Phase 6 feedback loop):
> - **Qdrant** (local Docker: `docker run -p 6333:6333 qdrant/qdrant`) — best for production nearest-neighbour search
> - **SQLite + numpy cosine sim** — zero-dependency, fine for development/small volume
>
> The plan implements both with a `USE_QDRANT` toggle in `config.py`. No decision needed to proceed.

---

## Proposed Changes

### File Layout

```
Observability_Clusters/
├── src/
│   ├── __init__.py
│   ├── config.py        # API key, thresholds, model names, date window
│   ├── fetch.py         # Phase 0a: pull spans from Mistral Observability API
│   ├── normalize.py     # Phase 0b: raw span fields → internal schema
│   ├── scorer.py        # Phase 0c: problem scoring (latency z-score, errors)
│   ├── moderation.py    # Phase 1:  guardrail subcategory enrichment
│   ├── router.py        # Phase 2:  deterministic known-category routing
│   ├── cluster.py       # Phase 3+4: discovery clustering + auto-label
│   ├── subcluster.py    # Phase 5:  subclustering per parent category
│   ├── feedback.py      # Phase 6:  NN matching + rule promotion
│   ├── store.py         # SQLite schema + optional Qdrant adapter
│   └── pipeline.py      # CLI orchestration
├── data/
│   └── spans.db         # SQLite: normalised spans + cluster history
├── notebooks/
│   └── explore_clusters.ipynb
├── .env.example
└── requirements.txt
```

---

### Component 1 — `requirements.txt`

```
mistralai>=1.0
umap-learn
hdbscan
qdrant-client
numpy
pandas
python-dotenv
```

---

### Component 2 — `src/config.py`

Central configuration — no hardcoded values elsewhere:

```python
import os
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

# Fetch window (Mistral retains 30 days)
FETCH_FROM = date.today() - timedelta(days=7)
FETCH_TO   = date.today()
PAGE_SIZE  = 100

# Models
EMBED_MODEL    = "mistral-embed"
MODERATE_MODEL = "mistral-moderation-latest"
LABEL_MODEL    = "mistral-large-latest"

# Scoring thresholds
LATENCY_Z_THRESHOLD = 2.5
RAGAS_FAITH_FLOOR   = 0.6

# Clustering
HDBSCAN_MIN_CLUSTER_SIZE = 8
UMAP_N_COMPONENTS        = 10

# Vector store — set True if Qdrant Docker is running
USE_QDRANT = False
QDRANT_URL = "http://localhost:6333"
```

---

### Component 3 — `src/fetch.py` (Phase 0a)

Pulls spans from the Mistral Observability API with full pagination:

```python
from mistralai.client import Mistral
from src.config import MISTRAL_API_KEY, FETCH_FROM, FETCH_TO, PAGE_SIZE

def discover_schema(client: Mistral):
    """Print live span field names for debugging normalize.py mappings."""
    fields = client.beta.observability.spans.list_span_fields()
    print("Live span fields:", [f.name for f in fields.fields])
    return fields

def fetch_all_spans(filters=None) -> list[dict]:
    spans = []
    with Mistral(api_key=MISTRAL_API_KEY) as client:
        discover_schema(client)
        page = 0
        while True:
            result = client.beta.observability.spans.search_spans(
                page_size=PAGE_SIZE,
                page=page,
                from_=FETCH_FROM,
                to_=FETCH_TO,
                filters=filters,
            )
            batch = result.data or []
            spans.extend([s.model_dump() for s in batch])
            if len(batch) < PAGE_SIZE:
                break
            page += 1
    return spans

def fetch_span_evaluations(span_ids: list[str]) -> dict:
    """Fetch quality eval scores (faithfulness etc.) for a list of span IDs."""
    evals = {}
    with Mistral(api_key=MISTRAL_API_KEY) as client:
        result = client.beta.observability.spans.search_span_evaluations(
            span_ids=span_ids
        )
        for ev in (result.data or []):
            evals[ev.span_id] = ev.scores  # dict of metric→score
    return evals
```

---

### Component 4 — `src/normalize.py` (Phase 0b)

Maps raw Mistral span dict → internal schema.
**Field paths are cross-checked against `list_span_fields()` output on first run:**

```python
from datetime import datetime

def parse_duration_ms(raw: dict) -> float | None:
    start, end = raw.get("start_time"), raw.get("end_time")
    if not (start and end):
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() * 1000
    except Exception:
        return None

def normalize_span(raw: dict, eval_scores: dict = None) -> dict:
    attrs = raw.get("attributes") or {}
    span_id = raw.get("span_id") or raw.get("id")
    scores = (eval_scores or {}).get(span_id, {})
    return {
        "span_id":        span_id,
        "trace_id":       raw.get("trace_id"),
        "operation_name": raw.get("name") or raw.get("operation_type"),
        "model":          attrs.get("model") or raw.get("model"),
        "tool_called":    attrs.get("tool_name"),
        "duration_ms":    parse_duration_ms(raw),
        "latency_zscore": None,           # set by scorer.py
        "status":         (raw.get("status") or "ok").lower(),
        "error_message":  raw.get("status_message") or attrs.get("error"),
        "exception_type": attrs.get("exception_type"),
        "input_summary":  str(attrs.get("input", ""))[:200],
        "output_summary": str(attrs.get("output", ""))[:200],
        "input_tokens":   attrs.get("input_tokens"),
        "output_tokens":  attrs.get("output_tokens"),
        "ragas_faithfulness": scores.get("faithfulness"),
        "retry_count":    int(attrs.get("retry_count") or 0),
        "timestamp":      raw.get("start_time"),
        # Phase 1 fields (filled later)
        "moderation_flagged":      False,
        "guardrail_subcategories": [],
        "guardrail_scores":        {},
        # Phase 2 fields (filled later)
        "parent_category": None,
        "subcategory":     None,
    }
```

---

### Component 5 — `src/scorer.py` (Phase 0c)

```python
import numpy as np
from collections import defaultdict
from src.config import LATENCY_Z_THRESHOLD, RAGAS_FAITH_FLOOR

def compute_latency_baselines(spans: list[dict]) -> dict:
    buckets = defaultdict(list)
    for s in spans:
        if s["duration_ms"] is not None:
            key = (s["model"] or "?", s["operation_name"] or "?")
            buckets[key].append(s["duration_ms"])
    return {
        k: {"mean": np.mean(v), "std": max(np.std(v), 1.0)}
        for k, v in buckets.items() if len(v) >= 5
    }

def score_spans(spans: list[dict], baselines: dict) -> list[dict]:
    for span in spans:
        key = (span["model"] or "?", span["operation_name"] or "?")
        b = baselines.get(key)
        span["latency_zscore"] = (
            (span["duration_ms"] - b["mean"]) / b["std"] if b and span["duration_ms"] else 0.0
        )
        flags = {
            "latency":   span["latency_zscore"] > LATENCY_Z_THRESHOLD,
            "quality":   (span["ragas_faithfulness"] or 1.0) < RAGAS_FAITH_FLOOR,
            "guardrail": span["moderation_flagged"],
            "error":     span["status"] not in ("ok", "success"),
            "retries":   span["retry_count"] > 0,
        }
        span["is_problematic"] = any(flags.values())
        span["flags"] = flags
    return spans
```

---

### Component 6 — `src/moderation.py` (Phase 1)

```python
from mistralai.client import Mistral
from src.config import MISTRAL_API_KEY, MODERATE_MODEL

def enrich_guardrail_spans(spans: list[dict]) -> list[dict]:
    # Pre-screen: only call moderation on error or flagged spans to save quota
    candidates = [s for s in spans if s["flags"]["error"] or s["flags"]["guardrail"]]
    with Mistral(api_key=MISTRAL_API_KEY) as client:
        for span in candidates:
            text = span["input_summary"] or span["output_summary"]
            if not text:
                continue
            resp = client.classifiers.moderate_chat(
                model=MODERATE_MODEL,
                inputs=[{"role": "user", "content": text}],
            )
            r = resp.results[0]
            if any(r.categories.values()):
                span["moderation_flagged"]      = True
                span["flags"]["guardrail"]      = True
                span["is_problematic"]          = True
                span["guardrail_subcategories"] = [
                    cat for cat, v in r.categories.items() if v
                ]
                span["guardrail_scores"]        = r.category_scores
    return spans
```

---

### Component 7 — `src/router.py` (Phase 2)

```python
def build_cluster_text(span: dict) -> str:
    parts = [
        span.get("error_message") or "",
        span.get("exception_type") or "",
        span.get("input_summary") or "",
        span.get("output_summary") or "",
        f"model={span.get('model')} op={span.get('operation_name')} tool={span.get('tool_called')}",
    ]
    return " | ".join(p for p in parts if p)

def classify_span(span: dict, store=None, embed_fn=None) -> str:
    if span["flags"]["guardrail"]:
        return "guardrail"
    if span["flags"]["latency"]:
        return "latency"
    if span["flags"]["quality"]:
        return "hallucination"
    if span["flags"]["error"]:
        return span.get("operation_name") or "tool_error"
    # Phase 6a: nearest-neighbour match against discovered clusters
    if store and embed_fn:
        vec = embed_fn(build_cluster_text(span))
        label = store.match(vec, threshold=0.85)
        if label:
            return label
    return "unclassified"
```

---

### Component 8 — `src/cluster.py` (Phase 3 + 4)

Weekly batch — embeds unclassified spans, clusters, auto-labels with Mistral:

```python
import numpy as np, umap, hdbscan, json
from mistralai.client import Mistral
from src.config import MISTRAL_API_KEY, EMBED_MODEL, LABEL_MODEL
from src.config import HDBSCAN_MIN_CLUSTER_SIZE, UMAP_N_COMPONENTS
from src.router import build_cluster_text

def run_discovery_clustering(spans: list[dict]):
    texts = [build_cluster_text(s) for s in spans]
    with Mistral(api_key=MISTRAL_API_KEY) as client:
        vectors = []
        for i in range(0, len(texts), 512):
            r = client.embeddings.create(model=EMBED_MODEL, inputs=texts[i:i+512])
            vectors.extend([e.embedding for e in r.data])

    arr = np.array(vectors)
    reduced = umap.UMAP(
        n_neighbors=15, min_dist=0.0,
        n_components=UMAP_N_COMPONENTS, metric="cosine"
    ).fit_transform(arr)

    labels = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        metric="euclidean", cluster_selection_method="eom"
    ).fit_predict(reduced)

    cluster_meta = {}
    with Mistral(api_key=MISTRAL_API_KEY) as client:
        for cid in set(labels):
            if cid == -1:
                continue
            members = [spans[i] for i, l in enumerate(labels) if l == cid]
            samples = "\n".join(f"- {build_cluster_text(s)}" for s in members[:8])
            resp = client.chat.complete(
                model=LABEL_MODEL, temperature=0,
                messages=[{"role": "user", "content":
                    f"Analyse {len(members)} failed AI pipeline spans:\n{samples}\n"
                    'Short category name (3-5 words) + one-sentence description.'
                    ' JSON: {"name": "...", "description": "..."}'
                }],
            )
            meta = json.loads(resp.choices[0].message.content)
            centroid = arr[[i for i, l in enumerate(labels) if l == cid]].mean(axis=0)
            cluster_meta[cid] = {**meta, "centroid": centroid.tolist(), "size": len(members)}

    assignments = {spans[i]["span_id"]: int(labels[i]) for i in range(len(spans))}
    return assignments, cluster_meta
```

---

### Component 9 — `src/subcluster.py` (Phase 5)

Same loop as `cluster.py` but scoped to one parent category, using a category-specific text builder:

```python
def build_latency_text(span):
    return (f"duration={span['duration_ms']:.0f}ms "
            f"op={span['operation_name']} model={span['model']}")

def build_error_text(span):
    return f"{span['exception_type']} | {span['error_message']} | op={span['operation_name']}"

CATEGORY_TEXT_FN = {
    "latency":      build_latency_text,
    "tool_error":   build_error_text,
    "hallucination": lambda s: f"faithfulness={s['ragas_faithfulness']} {s['output_summary']}",
}
```

Run as: `python -m src.pipeline subcluster --category latency`

---

### Component 10 — `src/pipeline.py` (Orchestration)

```
python -m src.pipeline ingest                 # fetch + score + route
python -m src.pipeline ingest --from 2026-09-03 --to 2026-09-10
python -m src.pipeline cluster                # weekly discovery batch
python -m src.pipeline subcluster --category latency
python -m src.pipeline report                 # print cluster summary table
```

---

## Build Order

| Step | Deliverable | Mistral APIs consumed | ~Effort |
|---|---|---|---|
| **1** | `fetch.py` + `normalize.py` + `store.py` | `spans.search_spans`, `list_span_fields`, `search_span_evaluations` | 1 day |
| **2** | `scorer.py` + `pipeline.py ingest` | none (local compute) | 0.5 day |
| **3** | `moderation.py` | `classifiers.moderate_chat` | 0.5 day |
| **4** | `router.py` | none | 0.5 day |
| **5** | `cluster.py` + `pipeline.py cluster` | `embeddings.create`, `chat.complete` | 1–2 days |
| **6** | `subcluster.py` | `embeddings.create`, `chat.complete` | 1 day |
| **7** | `feedback.py` + `pipeline.py report` | Qdrant or SQLite | 0.5 day |

---

## Verification Plan

### Automated Tests
```bash
pytest src/tests/test_fetch.py      # mock Mistral SDK, test pagination
pytest src/tests/test_scorer.py     # unit test z-score + flag logic
pytest src/tests/test_router.py     # routing precedence
pytest src/tests/test_cluster.py    # 50-span fixture, no API calls
```

### Manual Verification
1. **Schema discovery**: Run `discover_schema()` — print and verify field names match `normalize.py` mappings
2. **Ingest**: `pipeline.py ingest` → inspect `spans.db` — span count should match Studio Trace Explorer
3. **Scoring**: Check flag distribution in DB — expect 5–20% `is_problematic = True`
4. **Cluster**: `pipeline.py cluster` → human-review top 5 examples per cluster label for coherence
5. **Feedback**: Re-run ingest next day — verify NN matching routes known-bad patterns without reclustering
