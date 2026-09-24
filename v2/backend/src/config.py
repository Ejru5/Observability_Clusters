"""Central configuration for the v2 trace-level pipeline."""

import os
from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv

# Use the root-level .env (Observability_Clusters/.env)
_ROOT_ENV = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(_ROOT_ENV)

# ---------------------------------------------------------------------------
# Mistral API
# ---------------------------------------------------------------------------
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL: str = os.environ.get("MISTRAL_BASE_URL", "")

EMBED_MODEL  = "mistral-embed"
FACET_MODEL  = "mistral-large-latest"
LABEL_MODEL  = "mistral-large-latest"

MISTRAL_CLIENT_KWARGS = (
    {"server_url": MISTRAL_BASE_URL} if MISTRAL_BASE_URL else {}
)

# ---------------------------------------------------------------------------
# Fetch window
# ---------------------------------------------------------------------------
FETCH_FROM = date.today() - timedelta(days=15)
FETCH_TO   = date.today()
PAGE_SIZE  = 100

# ---------------------------------------------------------------------------
# Triage — bifurcation gate
# ---------------------------------------------------------------------------

# Hard-rule latency threshold percentile
TRIAGE_LATENCY_PERCENTILE = 95

# Autoencoder (high-recall tuned)
TRIAGE_AE_EPOCHS          = 200
TRIAGE_AE_LR              = 1e-3
TRIAGE_AE_HIDDEN_DIM      = 32
TRIAGE_AE_BOTTLENECK      = 8      # 10 input features → wider bottleneck
TRIAGE_AE_THRESHOLD_PCT   = 75    # LOW threshold = high recall (flag top 25%)

# ---------------------------------------------------------------------------
# Facet extraction
# ---------------------------------------------------------------------------
FACET_BATCH_SIZE = 10              # traces sent to LLM at once
FACET_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Cluster traces
# ---------------------------------------------------------------------------
# Cosine distance threshold for agglomerative bucketing
# distance = 1 - cosine_similarity
# 0.20 → traces must have similarity > 0.80 to cluster together
CLUSTER_DISTANCE_THRESHOLD = 0.20

# Minimum cluster size — 1 = keep singletons as Unique Errors
CLUSTER_MIN_SIZE = 1
