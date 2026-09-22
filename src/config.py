"""Central configuration -- edit here to tune fetch window, models, and clustering params."""

import os
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Mistral API
# ---------------------------------------------------------------------------
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL: str = os.environ.get("MISTRAL_BASE_URL", "")

# Models
EMBED_MODEL = "mistral-embed"
LABEL_MODEL = "mistral-large-latest"

# Mistral client kwargs -- server_url is forwarded when non-empty
MISTRAL_CLIENT_KWARGS = (
    {"server_url": MISTRAL_BASE_URL} if MISTRAL_BASE_URL else {}
)

# ---------------------------------------------------------------------------
# Fetch window  (Mistral retains 30 days max)
# ---------------------------------------------------------------------------
FETCH_FROM = date.today() - timedelta(days=15)
FETCH_TO   = date.today()
PAGE_SIZE  = 100

# ---------------------------------------------------------------------------
# Macro clustering  (run once across ALL spans)
# ---------------------------------------------------------------------------
MACRO_UMAP = dict(
    n_neighbors=15,
    min_dist=0.0,
    n_components=10,
    metric="cosine",
    random_state=42,
)
MACRO_HDBSCAN = dict(
    min_cluster_size=3,      # reduced: input is now only the problematic subset
    metric="euclidean",
    cluster_selection_method="eom",
)

# ---------------------------------------------------------------------------
# Micro clustering  (run independently per macro cluster)
# ---------------------------------------------------------------------------
MICRO_UMAP = dict(
    n_neighbors=10,
    min_dist=0.0,
    n_components=5,
    metric="cosine",
    random_state=42,
)
MICRO_HDBSCAN = dict(
    min_cluster_size=2,      # lowered so macro clusters of size 4+ can subcluster
    metric="euclidean",
    cluster_selection_method="eom",
)

# Minimum number of spans a macro cluster must have before we attempt subclustering
MICRO_MIN_SPANS = 4

# ---------------------------------------------------------------------------
# Triage — Bifurcation step (rules + LOF + Autoencoder)
# ---------------------------------------------------------------------------

# Active algorithm used to write problematic_spans.jsonl
# "lof" | "ae"  — toggled by POST /api/triage/select via the UI
TRIAGE_ALGORITHM = "lof"

# Rule-based: percentile threshold for high-latency flag
TRIAGE_LATENCY_PERCENTILE = 95

# LOF settings (runs on UMAP-reduced embeddings)
TRIAGE_LOF_N_NEIGHBORS   = 20
TRIAGE_LOF_CONTAMINATION = 0.10   # fraction of spans expected to be anomalous

# Autoencoder settings (PyTorch MLP, runs on structured feature matrix)
TRIAGE_AE_EPOCHS         = 150    # training epochs
TRIAGE_AE_LR             = 1e-3   # Adam learning rate
TRIAGE_AE_HIDDEN_DIM     = 32     # hidden layer width
TRIAGE_AE_BOTTLENECK     = 6      # bottleneck (latent) size
TRIAGE_AE_THRESHOLD_PCT  = 95     # flag spans above this percentile of recon error
