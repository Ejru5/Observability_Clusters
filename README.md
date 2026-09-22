# Observability Clusters — LLM Trace Anomaly Detection & Hierarchical Clustering

A **production-ready observability pipeline** that fetches LLM telemetry spans from the Mistral Observability API, detects problematic traces using both deterministic rules and unsupervised ML (LOF + Autoencoder), then clusters them into a two-level semantic hierarchy using UMAP + HDBSCAN — all without rule-based routing.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Installation & Setup](#5-installation--setup)
6. [Configuration](#6-configuration)
7. [Pipeline Reference](#7-pipeline-reference)
8. [Triage Algorithms](#8-triage-algorithms)
9. [Clustering Algorithms](#9-clustering-algorithms)
10. [API Reference](#10-api-reference)
11. [Frontend Dashboard (UI)](#11-frontend-dashboard-ui)
12. [Data Files Reference](#12-data-files-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Advanced Guides](#14-advanced-guides)

---

## 1. Overview

This project solves the problem of making sense of large volumes of LLM observability data from Mistral-powered AI pipelines. Given thousands of telemetry spans (individual LLM calls, tool invocations, agent steps), it:

1. **Fetches** raw spans from the Mistral Observability API over a configurable date window
2. **Normalises** them into a consistent schema and embeds them using `mistral-embed`
3. **Triages** spans into *healthy* vs *problematic* using three complementary methods:
   - Deterministic hard rules (API errors, high latency, truncated output, etc.)
   - **Local Outlier Factor (LOF)** on UMAP-reduced embeddings
   - **PyTorch Autoencoder** on a structured feature matrix
4. **Clusters** the problematic subset at two levels:
   - **Macro level**: broad failure-mode categories (latency issues, API errors, tool failures, etc.)
   - **Micro level**: fine-grained subclusters within each macro category
5. **Labels** each cluster semantically using `mistral-large-latest`
6. **Exposes** results via a FastAPI REST backend and a Next.js interactive dashboard

---

## 2. Architecture

```
+-----------------------------------------------------------------------------------+
|                          Mistral Observability API                                |
+-----------------------------------------------------------------------------------+
                                     |  fetch_spans.py
                                     v
                          data/raw_spans.jsonl
                                     |  embed.py
                                     v
              data/spans.jsonl + data/embeddings.npy
                                     |  triage.py
                    +----------------+------------------+
                    v                                   v
          [Phase 1 -- Hard Rules]          [Phase 2a -- LOF (UMAP)]
          - status == "error"              [Phase 2b -- Autoencoder]
          - exception raised
          - HTTP >= 400                      data/triage_scores.json
          - high latency P95
          - truncated output
          - empty response
                    +----------------+------------------+
                                     |  apply_triage.py  (selectable: lof | ae)
                                     v
              data/problematic_spans.jsonl
              data/problematic_embeddings.npy
                                     |  cluster_macro.py
                                     v
                    UMAP (10D) -> HDBSCAN
                         data/macro_clusters.json
                                     |  cluster_micro.py
                                     v
              per-macro UMAP (5D) -> HDBSCAN
                         data/micro_clusters.json
                                     |  label.py
                                     v
            mistral-large-latest auto-labelling
           (heuristic fallback when API unavailable)
                                     |  report.py
                                     v
                         data/report.md

              +-------------------+--+--------------------+
              v                                            v
    FastAPI backend (api/main.py)            Next.js dashboard (ui/)
    uvicorn :8000                            next dev :3000
```

### Two-Level Clustering Design

The macro and micro clustering passes are computed **independently**, not hierarchically in one embedding space. This separation is intentional:

- **Macro pass**: distinguishes *failure categories* from each other (latency vs. API error vs. tool failure), using the full span content
- **Micro pass**: runs independently *within* each macro cluster, on the same embeddings but with its own UMAP/HDBSCAN parameters tuned for finer granularity

Mixing both into a single global embedding compresses two different signals into one representation and does neither well.

---

## 3. Project Structure

```
Observability_Clusters/
+-- .env                          # Local secrets (not committed)
+-- .env.example                  # Template for required env vars
+-- requirements.txt              # Python dependencies
|
+-- src/                          # Core pipeline modules
|   +-- config.py                 # Central configuration (all tunable params)
|   +-- fetch_spans.py            # Step 1: Fetch spans from Mistral API
|   +-- embed.py                  # Step 2: Normalise + embed with mistral-embed
|   +-- triage.py                 # Step 2.5: LOF + Autoencoder anomaly detection
|   +-- apply_triage.py           # Step 2.6: Filter spans by chosen algorithm
|   +-- cluster_macro.py          # Step 3: Top-level UMAP + HDBSCAN clustering
|   +-- cluster_micro.py          # Step 4: Per-macro sub-clustering
|   +-- label.py                  # Step 5: LLM auto-labelling (+ heuristic fallback)
|   +-- report.py                 # Step 6: Human-readable markdown report
|
+-- api/
|   +-- main.py                   # FastAPI REST backend
|
+-- ui/                           # Next.js 16 + TypeScript frontend
|   +-- app/
|   |   +-- page.tsx              # Main dashboard (single-page app)
|   |   +-- layout.tsx            # Root layout
|   |   +-- globals.css           # Global styles + Tailwind
|   +-- package.json
|   +-- next.config.ts
|
+-- data/                         # Generated data files (git-ignored)
|   +-- raw_spans.jsonl
|   +-- spans.jsonl
|   +-- embeddings.npy
|   +-- triage_scores.json
|   +-- triage_report.json
|   +-- problematic_spans.jsonl
|   +-- problematic_embeddings.npy
|   +-- macro_clusters.json
|   +-- micro_clusters.json
|   +-- points_2d.json
|   +-- report.md
|
+-- autoencoder_optimization_guide.md   # Deep-dive: AE architecture + accuracy
+-- mistral-trace-clustering-guide.md   # Design reference for hierarchical clustering
```

---

## 4. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11+ recommended |
| Node.js | 18+ | For the Next.js UI |
| Mistral API Key | -- | With Observability API access |
| CUDA (optional) | -- | GPU acceleration for the Autoencoder |

---

## 5. Installation & Setup

### 5.1 Python Backend

```bash
# Clone and enter the project
cd Observability_Clusters

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
mistralai>=1.0
umap-learn
hdbscan
numpy
python-dotenv
scikit-learn
torch
```

> **Note on PyTorch**: The above installs the CPU build by default. For GPU support,
> follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
> for your CUDA version.

### 5.2 Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:
```env
MISTRAL_API_KEY=your_api_key_here
MISTRAL_BASE_URL=https://api.f418c72b5f65.dc.mistral.ai
```

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | Your Mistral API key with Observability access |
| `MISTRAL_BASE_URL` | Optional | Custom endpoint (e.g., on-prem or regional). Leave empty to use the public API. |

### 5.3 FastAPI Backend

```bash
# From the project root (with .venv active)
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

### 5.4 Next.js Frontend

```bash
cd ui
npm install
npm run dev
```

The dashboard will be available at `http://localhost:3000`.

---

## 6. Configuration

All tunable parameters live in `src/config.py`. Edit this file to change behaviour
without modifying any pipeline code.

### 6.1 Fetch Window

```python
FETCH_FROM = date.today() - timedelta(days=15)  # Start of fetch window
FETCH_TO   = date.today()                         # End of fetch window
PAGE_SIZE  = 100                                  # Spans per API page
```

> Mistral Observability retains data for a maximum of 30 days.

### 6.2 Models

```python
EMBED_MODEL  = "mistral-embed"         # Used in embed.py
LABEL_MODEL  = "mistral-large-latest"  # Used in label.py
```

### 6.3 Macro Clustering Parameters

```python
MACRO_UMAP = dict(
    n_neighbors=15,     # Higher = more global structure preserved
    min_dist=0.0,       # Lower = tighter clusters (ideal for HDBSCAN)
    n_components=10,    # Dimensionality after reduction
    metric="cosine",
    random_state=42,
)

MACRO_HDBSCAN = dict(
    min_cluster_size=3,             # Raise if too many tiny clusters
    metric="euclidean",
    cluster_selection_method="eom", # "eom" (excess of mass) or "leaf"
)
```

### 6.4 Micro Clustering Parameters

```python
MICRO_UMAP = dict(
    n_neighbors=10,
    min_dist=0.0,
    n_components=5,
    metric="cosine",
    random_state=42,
)

MICRO_HDBSCAN = dict(
    min_cluster_size=2,  # Lower = finer subclusters
    metric="euclidean",
    cluster_selection_method="eom",
)

MICRO_MIN_SPANS = 4  # Minimum macro cluster size to attempt subclustering
```

### 6.5 Triage Algorithm Selection

```python
TRIAGE_ALGORITHM = "lof"  # "lof" | "ae" -- toggled via UI or apply_triage --algo
```

### 6.6 LOF Parameters

```python
TRIAGE_LOF_N_NEIGHBORS   = 20    # k-neighbours for local outlier score
TRIAGE_LOF_CONTAMINATION = 0.10  # Expected fraction of anomalous spans (0.0 to 0.5)
```

### 6.7 Autoencoder Parameters

```python
TRIAGE_AE_EPOCHS        = 150   # Training iterations
TRIAGE_AE_LR            = 1e-3  # Adam learning rate
TRIAGE_AE_HIDDEN_DIM    = 32    # Width of encoder entry layer (expands to hidden*2 internally)
TRIAGE_AE_BOTTLENECK    = 6     # Latent space dimensions (8 -> 6 compression)
TRIAGE_AE_THRESHOLD_PCT = 95    # Flag spans above this reconstruction error percentile
```

### 6.8 Triage Latency Rule

```python
TRIAGE_LATENCY_PERCENTILE = 95  # Spans with duration_ms > P95 are flagged as high_latency
```

---

## 7. Pipeline Reference

Run each step independently in order. Steps are idempotent — re-running an existing
step will overwrite its output files.

### Step 1 -- Fetch Spans

```bash
python -m src.fetch_spans

# Override date window:
python -m src.fetch_spans --from 2026-09-01 --to 2026-09-20
```

**Input**: Mistral Observability API
**Output**: `data/raw_spans.jsonl`

Cursor-paginates through all spans in the configured date window and saves raw API
responses (one JSON object per line). Also discovers and prints live schema field names.

---

### Step 2 -- Embed

```bash
python -m src.embed
```

**Input**: `data/raw_spans.jsonl`
**Output**: `data/spans.jsonl`, `data/embeddings.npy`

Normalises raw spans into a clean internal schema, builds a rich text representation
for each span, and embeds them in batches of 64 using `mistral-embed`.

**Normalised span fields:**

| Field | Description |
|---|---|
| `span_id` | Unique identifier |
| `trace_id` | Parent trace ID |
| `operation_name` | e.g., `chat`, `completion`, `tool_call` |
| `model` | LLM model used |
| `tool_called` | Tool name if this is a tool span |
| `duration_ms` | Derived from `duration_ns` or start/end timestamps |
| `status` | `ok`, `error`, `unset` |
| `error_message` | Error/status message if any |
| `exception_type` | Python exception type if raised |
| `input_summary` | First 200 chars of input messages |
| `output_summary` | First 200 chars of output messages |
| `input_tokens` | Token count for input |
| `output_tokens` | Token count for output |
| `timestamp` | Span start time (ISO 8601) |

---

### Step 2.5 -- Triage

```bash
python -m src.triage
```

**Input**: `data/spans.jsonl`, `data/embeddings.npy`, `data/raw_spans.jsonl`
**Output**: `data/triage_scores.json`, `data/triage_report.json`

Runs two phases (see [Triage Algorithms](#8-triage-algorithms) for full details):

- **Phase 1 -- Hard rules**: deterministic flags for known failure patterns
- **Phase 2a -- LOF**: Local Outlier Factor on UMAP-reduced embeddings
- **Phase 2b -- Autoencoder**: PyTorch MLP reconstruction error on structured features

Both ML results are saved together so the UI can switch between them without re-running.

---

### Step 2.6 -- Apply Triage

```bash
python -m src.apply_triage --algo lof   # or --algo ae
```

**Input**: `data/triage_scores.json`, `data/spans.jsonl`, `data/embeddings.npy`
**Output**: `data/problematic_spans.jsonl`, `data/problematic_embeddings.npy`

Filters the full span set down to only the problematic subset (rule-fired OR ML-flagged
by the chosen algorithm). Attaches triage metadata to each span and invalidates the
`points_2d.json` cache so the dashboard re-computes on next load.

---

### Step 3 -- Macro Clustering

```bash
python -m src.cluster_macro
```

**Input**: `data/problematic_spans.jsonl`, `data/problematic_embeddings.npy`
**Output**: `data/macro_clusters.json`

Reduces the problematic embedding space with UMAP (10D), then runs HDBSCAN to produce
top-level failure-mode clusters. Stores each cluster's centroid, size, span IDs, and
a noise flag.

---

### Step 4 -- Micro Clustering

```bash
python -m src.cluster_micro
```

**Input**: `data/problematic_spans.jsonl`, `data/problematic_embeddings.npy`, `data/macro_clusters.json`
**Output**: `data/micro_clusters.json`

For each macro cluster with >= `MICRO_MIN_SPANS` members, independently runs
UMAP (5D) + HDBSCAN with finer parameters to discover subclusters within the parent
failure category.

---

### Step 5 -- Label

```bash
python -m src.label
```

**Input**: `data/macro_clusters.json`, `data/micro_clusters.json`, `data/spans.jsonl`
**Output**: Both cluster JSON files updated in-place with `label` and `description` fields

Samples up to 8 representative spans per cluster and uses `mistral-large-latest`
(zero temperature) to generate a concise 3-5 word semantic failure title and a
one-sentence description. Falls back to a deterministic heuristic labeller if the
API is unavailable.

**Label contract:**
- Names describe the *failure mode*, not just the operation
  (e.g., `"Truncated Travel Itinerary Responses"` not `"Chat Completion"`)
- Noise clusters are always labeled `[NOISE]`

---

### Step 6 -- Report

```bash
python -m src.report

# Show span examples:
python -m src.report --examples 3
```

**Input**: `data/macro_clusters.json`, `data/micro_clusters.json`, `data/spans.jsonl`
**Output**: `data/report.md` (also printed to stdout)

Generates a human-readable two-level tree showing all macro clusters, their
subclusters, sizes, labels, and optionally sample spans with error messages and
input/output previews.

---

### Running the Full Pipeline

```bash
python -m src.fetch_spans
python -m src.embed
python -m src.triage
python -m src.apply_triage --algo lof
python -m src.cluster_macro
python -m src.cluster_micro
python -m src.label
python -m src.report
```

Or trigger it all via the API (runs in background, with progress tracking):

```bash
curl -X POST http://localhost:8000/api/pipeline/run-all
curl http://localhost:8000/api/pipeline/full-status
```

---

## 8. Triage Algorithms

Triage bifurcates spans into **healthy** vs **problematic** before clustering.
Only problematic spans are passed to clustering, keeping cluster quality high.

### Phase 1 -- Hard Rules (always active, deterministic)

| Rule | Condition | Tag |
|---|---|---|
| API error status | `status == "error"` | `api_error` |
| Exception raised | `exception_type` is non-empty | `exception_error` / `tool_error` |
| Non-empty error message | `error_message` is set | `api_error` |
| HTTP 4xx/5xx | `http.response.status_code >= 400` | `api_error` |
| Truncated output | `finish_reason == "length"` | `truncated_output` |
| High latency | `duration_ms > P95` of all spans | `high_latency` |
| Empty response | `output_tokens == 0` on chat/completion | `empty_response` |

### Phase 2a -- Local Outlier Factor (LOF)

Runs on UMAP-reduced embeddings (same parameters as macro clustering). LOF computes
a local density ratio for each point relative to its k-nearest neighbours. Points in
sparse regions get high outlier scores.

- **Contamination**: controlled by `TRIAGE_LOF_CONTAMINATION` (default `0.10`)
- **Score**: higher = more anomalous
- **Label**: `-1` for anomaly, `1` for normal

### Phase 2b -- Autoencoder (AE)

A PyTorch MLP trained on a structured 8-column feature matrix:

| Column | Feature | Normalization |
|---|---|---|
| 0 | `duration_ms` | Min-max |
| 1 | `input_tokens` | Min-max |
| 2 | `output_tokens` | Min-max |
| 3 | `output_token_ratio` = output / (input + output + 1) | Min-max |
| 4 | `is_chat_span` (op in {chat, completion, fim}) | Binary |
| 5 | `is_tool_span` | Binary |
| 6 | `has_model` | Binary |
| 7 | `has_error_message` | Binary |

**Architecture**: `8 -> 32 -> 64 -> 6 (bottleneck) -> 64 -> 32 -> 8`

Spans whose reconstruction MSE exceeds the `P{TRIAGE_AE_THRESHOLD_PCT}` percentile
are flagged as anomalous.

### Switching Algorithms

The triage step computes **both** LOF and AE results and saves them together.
The `apply_triage` step selects which result to use:

```bash
python -m src.apply_triage --algo lof
python -m src.apply_triage --algo ae
```

The UI provides a live toggle that calls `POST /api/triage/select`, which re-runs
`apply_triage` + clustering + labelling in the background.

### Combined Logic

A span is considered problematic if:
- **Hard rules fired** (always applicable), **OR**
- **ML algorithm flagged it** (LOF or AE, depending on selection)

Problem type tags:
- `ml_anomaly`: only ML flagged it (no rule fired)
- `ml_confirmed`: both ML and a hard rule flagged it

---

## 9. Clustering Algorithms

### UMAP (Uniform Manifold Approximation and Projection)

Reduces the high-dimensional embedding space (typically 1024D from `mistral-embed`)
to a lower-dimensional manifold for density-based clustering:

- **Macro**: 10 components, 15 neighbours, cosine metric
- **Micro**: 5 components, 10 neighbours, cosine metric (run per macro cluster)

`min_dist=0.0` packs points tightly, ideal for HDBSCAN density estimation.

### HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

Density-based clustering that finds clusters of arbitrary shape and marks
low-density points as noise (`cluster_id = -1`):

- **Macro**: `min_cluster_size=3`, `cluster_selection_method="eom"`
- **Micro**: `min_cluster_size=2`, `cluster_selection_method="eom"`

Advantages over k-means: no need to specify number of clusters; naturally handles
noise; robust to outliers; captures non-spherical cluster shapes.

### 2D Visualisation

For the scatter plot dashboard view, embeddings are further projected to 2D
(UMAP with `n_components=2`, `min_dist=0.1`), falling back to PCA if UMAP fails.
Coordinates are normalised to [-100, 100] and cached in `data/points_2d.json`.
This cache is invalidated whenever the triage algorithm is switched.

---

## 10. API Reference

Interactive docs: `http://localhost:8000/docs`

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check -- returns `{"status": "ok"}` |

### Pipeline

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/status` | Per-step completion status (file existence check) |
| `POST` | `/api/pipeline/run` | Run a single step in background (`{"step": "fetch"}`) |
| `POST` | `/api/pipeline/run-all` | Run the entire pipeline end-to-end in background |
| `GET` | `/api/pipeline/full-status` | Progress, current step, per-step statuses, errors, duration |

Valid step names: `fetch`, `embed`, `triage`, `apply_triage`, `macro`, `micro`, `label`, `report`

### Clusters

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/clusters` | List all macro clusters (sorted by size, noise last) |
| `GET` | `/api/clusters/{cluster_id}` | Full detail: label, description, span IDs, micro clusters |
| `GET` | `/api/cluster-points` | 2D scatter data for all problematic spans (UMAP projection) |

### Spans

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/spans` | Paginated span list (`?limit=100&offset=0`) |
| `GET` | `/api/spans/{span_id}` | Single span detail with full triage fields |
| `GET` | `/api/clusters/{cluster_id}/spans` | Spans for a specific cluster (`?limit=50`) |

### Triage

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/triage` | Aggregate triage report (counts, problem type breakdown) |
| `GET` | `/api/triage/scores` | Per-span LOF + AE scores (`?limit=200&offset=0`) |
| `POST` | `/api/triage/select` | Switch algorithm (`{"algorithm": "lof"}` or `"ae"`) and re-cluster |

### Stats & Report

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/stats` | Span count, cluster counts, model/op distribution |
| `GET` | `/api/report` | Markdown cluster report as plain text |

---

## 11. Frontend Dashboard (UI)

The Next.js dashboard (`ui/`) provides an interactive view of the clustering results.

### Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 16 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS v4 |
| Charts | Recharts |
| Icons | Phosphor Icons |
| Animation | Motion (Framer Motion successor) |

### Running the UI

```bash
cd ui
npm install
npm run dev     # Development server -> http://localhost:3000
npm run build   # Production build
npm run start   # Serve production build
```

### Dashboard Features

- **Pipeline control panel**: trigger individual steps or the full pipeline with real-time progress tracking
- **Cluster overview**: table of macro clusters sorted by size, with micro-cluster counts and labels
- **Scatter plot**: 2D UMAP projection of all problematic spans, colour-coded by cluster assignment
- **Cluster detail**: drill into any macro cluster to see its micro subclusters and member spans
- **Triage panel**: view LOF/AE anomaly scores, switch between algorithms live
- **Span inspector**: per-span view with full metadata, error messages, input/output previews, and triage tags

---

## 12. Data Files Reference

All generated files are stored in `data/` and are git-ignored.

| File | Format | Description |
|---|---|---|
| `raw_spans.jsonl` | JSONL | Raw Mistral API responses, one span object per line |
| `spans.jsonl` | JSONL | Normalised spans in the internal schema |
| `embeddings.npy` | NumPy float32 [N, dim] | Embedding vectors aligned row-for-row with `spans.jsonl` |
| `triage_scores.json` | JSON array | Per-span LOF scores, AE reconstruction errors, rule flags, and labels |
| `triage_report.json` | JSON | Aggregate triage statistics (counts by algorithm and problem type) |
| `problematic_spans.jsonl` | JSONL | Filtered subset -- only problematic spans, with triage metadata attached |
| `problematic_embeddings.npy` | NumPy float32 [M, dim] | Embeddings aligned with `problematic_spans.jsonl` |
| `macro_clusters.json` | JSON | Top-level clusters: label, description, span_ids, centroid, size, is_noise |
| `micro_clusters.json` | JSON | Nested: `{macro_id: {micro_id: cluster_record}}` |
| `points_2d.json` | JSON array | 2D UMAP scatter data for dashboard; deleted on algorithm switch |
| `report.md` | Markdown | Human-readable two-level cluster tree |

### Cluster Record Schema

```json
{
  "label":       "Truncated Travel Itinerary Responses",
  "description": "Cluster of spans where output was cut short due to token limits.",
  "span_ids":    ["abc123", "def456"],
  "centroid":    [0.12, -0.34, "..."],
  "size":        42,
  "is_noise":    false
}
```

---

## 13. Troubleshooting

### `MISTRAL_API_KEY is not set`
Ensure `.env` exists in the project root with your API key. The pipeline checks this
at startup and exits with a clear error message.

### `raw_spans.jsonl is empty` / `No spans returned`
- Verify your API key has Observability API access
- Check the `FETCH_FROM`/`FETCH_TO` window in `config.py` is within Mistral's 30-day retention
- Try a custom date range: `python -m src.fetch_spans --from 2026-09-01 --to 2026-09-10`

### `spans (N) and embeddings (M) are misaligned`
Re-run `python -m src.embed` -- this regenerates both `spans.jsonl` and `embeddings.npy` atomically.

### Too many tiny macro clusters
Raise `MACRO_HDBSCAN["min_cluster_size"]` in `config.py` (e.g., from `3` to `5` or `8`).

### Subclusters too coarse / no subclusters
Lower `MICRO_HDBSCAN["min_cluster_size"]` (e.g., from `2` to `1`) or lower `MICRO_MIN_SPANS`.

### LOF flags too many / too few spans
Adjust `TRIAGE_LOF_CONTAMINATION` in `config.py`:
- Too many false positives: lower it (e.g., `0.05`)
- Missing real anomalies: raise it (e.g., `0.15`)

### Autoencoder flags exactly 5% of spans every time
This is the static P95 threshold behaviour. See the [Autoencoder Optimization Guide](autoencoder_optimization_guide.md)
for dynamic IQR-based thresholding and other accuracy improvements.

### Labels are generic or not meaningful
Check that `MISTRAL_API_KEY` is valid and the chat completion API is accessible.
If the LLM is unavailable, the heuristic fallback produces simpler labels based on
the most common operation name and problem type in the cluster.

### FastAPI 404 on `/api/clusters`
Run the full pipeline first (at minimum steps 1-4: `fetch`, `embed`, `triage`,
`apply_triage`, `macro`, `micro`). The API returns 404 if required data files don't exist.

### `pip install torch` installs a very large package
For CPU-only usage:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 14. Advanced Guides

### Autoencoder Optimization Guide

`autoencoder_optimization_guide.md` documents:

- The exact anomaly detection mechanism (manifold learning + MSE reconstruction error)
- All hyperparameters with technical descriptions
- 5 identified accuracy bottlenecks in the current implementation:
  - Min-max scaling distortion on power-law distributed metrics
  - Training set contamination (AE trains on all spans including errors)
  - Under-compressed bottleneck (8 -> 6 is only 25% reduction)
  - Gradient disparity on mixed continuous/binary features
  - Static percentile cutoff artifacts
- 4 concrete improvement strategies (log-transforms, semi-supervised healthy-only training, calibrated bottleneck with BatchNorm/LeakyReLU/Dropout, dynamic IQR/3-sigma thresholding)
- Full drop-in code implementation for the optimized autoencoder

### Hierarchical Clustering Design Reference

`mistral-trace-clustering-guide.md` is the design document for the broader system, covering:

- Problem scoring and pre-filtering philosophy
- Guardrail subcategory extraction via Mistral moderation API
- Known-category rule layer design
- Discovery clustering workflow for unclassified spans
- Auto-labelling and centroid storage for new-span matching (Qdrant)
- Feedback loop: promoting stable clusters to deterministic rules
- Comparison with Arize Phoenix, Galileo Signals, Braintrust Topics, and Datadog Patterns

---

## Contributing

- All pipeline configuration belongs in `src/config.py` -- never hardcode values in step modules
- Each pipeline step must be runnable independently (check for its input files, exit clearly if missing)
- The `run = main` alias at the bottom of each module is required for the API's background subprocess runner
- If you add a new pipeline step, register it in `PIPELINE_ORDER` and `module_map` in `api/main.py`

---

*Built on [Mistral Observability API](https://docs.mistral.ai/capabilities/observability/) · Powered by `mistral-embed` + `mistral-large-latest` · Clustering via `umap-learn` + `hdbscan` · Anomaly detection via `scikit-learn` LOF + PyTorch AE*
