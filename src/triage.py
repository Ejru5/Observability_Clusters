"""
Step 2.5 -- Triage: bifurcate spans into healthy vs problematic.

Runs in two phases:
  Phase 1 — Hard rules (deterministic, always fire):
    - status == "error"
    - exception_type is non-empty
    - error_message is non-empty
    - http_status >= 400
    - finish_reason == "length"  (truncated output)
    - duration_ms > P95 of all spans  (high latency)
    - output_tokens == 0 on a chat/completion span  (empty response)

  Phase 2a — LOF on UMAP-reduced embeddings (unsupervised)
  Phase 2b — Autoencoder on structured feature matrix (unsupervised)

Both algorithm results are saved together so the UI toggle switches
between them without re-running this step.

Reads:   data/spans.jsonl
         data/embeddings.npy
         data/raw_spans.jsonl   (for http_status + finish_reason fields)
Writes:  data/triage_scores.json   (per-span scores from both algos + rules)
         data/triage_report.json   (aggregate stats / breakdown)

Usage:
    python -m src.triage
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from typing import Any

SPAN_FILE    = Path("data/spans.jsonl")
RAW_FILE     = Path("data/raw_spans.jsonl")
EMB_FILE     = Path("data/embeddings.npy")
SCORES_FILE  = Path("data/triage_scores.json")
REPORT_FILE  = Path("data/triage_report.json")

from src.config import (
    TRIAGE_LATENCY_PERCENTILE,
    TRIAGE_LOF_N_NEIGHBORS,
    TRIAGE_LOF_CONTAMINATION,
    TRIAGE_AE_EPOCHS,
    TRIAGE_AE_LR,
    TRIAGE_AE_HIDDEN_DIM,
    TRIAGE_AE_BOTTLENECK,
    TRIAGE_AE_THRESHOLD_PCT,
    MACRO_UMAP,
)


# ---------------------------------------------------------------------------
# Phase 1 — Hard rules
# ---------------------------------------------------------------------------

def _build_raw_index() -> dict[str, dict]:
    """Index raw spans by span_id to pull http_status + finish_reason."""
    if not RAW_FILE.exists():
        return {}
    idx: dict[str, dict] = {}
    for line in RAW_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            sid = r.get("span_id", "")
            if sid:
                idx[sid] = r
    return idx


def apply_rules(span: dict, raw: dict, p95_ms: float) -> tuple[bool, list[str]]:
    """
    Return (rule_fired, list_of_problem_types).
    Each problem type is a short string tag.
    """
    problem_types: list[str] = []

    status = (span.get("status") or "").lower()
    exc    = (span.get("exception_type") or "").strip()
    errmsg = (span.get("error_message") or "").strip()
    dur    = span.get("duration_ms")
    op     = (span.get("operation_name") or "").lower()
    out_t  = span.get("output_tokens")
    tool   = (span.get("tool_called") or "").strip()

    # Explicit error status
    if status == "error":
        problem_types.append("api_error")

    # Exception raised
    if exc:
        tag = "tool_error" if tool else "exception_error"
        if tag not in problem_types:
            problem_types.append(tag)

    # Non-empty error message (and not already flagged)
    if errmsg and "api_error" not in problem_types and "exception_error" not in problem_types:
        problem_types.append("api_error")

    # HTTP 4xx / 5xx from raw span_attributes
    attrs = raw.get("span_attributes") or {}
    http_code_str = str(
        attrs.get("http.response.status_code")
        or attrs.get("http.status_code")
        or ""
    )
    if http_code_str.isdigit() and int(http_code_str) >= 400:
        if "api_error" not in problem_types:
            problem_types.append("api_error")

    # Truncated output (finish_reason == "length")
    finish_reasons = raw.get("response_finish_reasons") or []
    if "length" in finish_reasons:
        problem_types.append("truncated_output")

    # High latency
    if dur is not None and p95_ms > 0 and dur > p95_ms:
        problem_types.append("high_latency")

    # Empty response on a generative span
    if op in ("chat", "completion", "fim") and (out_t == 0 or out_t is None):
        problem_types.append("empty_response")

    return bool(problem_types), problem_types


# ---------------------------------------------------------------------------
# Phase 2a — LOF on UMAP-reduced embeddings
# ---------------------------------------------------------------------------

def run_lof(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      lof_scores   float[N]  higher = more anomalous
      lof_labels   int[N]    -1 anomaly, 1 normal
    """
    try:
        import umap as umap_lib
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        sys.exit("ERROR: pip install umap-learn scikit-learn")

    n = len(embeddings)
    print(f"  LOF: reducing {n} embeddings via UMAP ...", end=" ", flush=True)

    n_comp  = min(MACRO_UMAP["n_components"], n - 2)
    n_nbrs  = min(MACRO_UMAP["n_neighbors"], n - 1)
    reducer = umap_lib.UMAP(
        n_components=n_comp,
        n_neighbors=n_nbrs,
        metric=MACRO_UMAP["metric"],
        min_dist=0.0,
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"-> {reduced.shape}")

    n_neighbors_lof = min(TRIAGE_LOF_N_NEIGHBORS, n - 1)
    print(f"  LOF: LocalOutlierFactor (n_neighbors={n_neighbors_lof}, contamination={TRIAGE_LOF_CONTAMINATION}) ...", end=" ", flush=True)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors_lof,
        contamination=TRIAGE_LOF_CONTAMINATION,
        algorithm="ball_tree",
        n_jobs=-1,
    )
    labels = lof.fit_predict(reduced)
    scores = -lof.negative_outlier_factor_
    n_anom = int((labels == -1).sum())
    print(f"-> {n_anom} anomalies flagged")

    return scores.astype(float), labels.astype(int)


# ---------------------------------------------------------------------------
# Phase 2b — Autoencoder on structured feature matrix
# ---------------------------------------------------------------------------

def _build_feature_matrix(spans: list[dict]) -> np.ndarray:
    """
    8-column numeric matrix, all values normalized 0-1.
    Columns:
      0  duration_ms normalized
      1  input_tokens normalized
      2  output_tokens normalized
      3  output_token_ratio  (output / (input + output + 1))
      4  is_chat_span
      5  is_tool_span
      6  has_model
      7  has_error_message
    """
    raw = []
    for s in spans:
        dur   = float(s.get("duration_ms") or 0.0)
        in_t  = float(s.get("input_tokens") or 0.0)
        out_t = float(s.get("output_tokens") or 0.0)
        op    = (s.get("operation_name") or "").lower()
        raw.append([
            dur,
            in_t,
            out_t,
            out_t / (in_t + out_t + 1.0),
            1.0 if op in ("chat", "completion", "fim") else 0.0,
            1.0 if s.get("tool_called") else 0.0,
            1.0 if s.get("model") else 0.0,
            1.0 if (s.get("error_message") or "").strip() else 0.0,
        ])
    arr = np.array(raw, dtype=np.float32)

    # Min-max normalize continuous cols (skip binary cols 4-7)
    for col in range(4):
        col_min, col_max = arr[:, col].min(), arr[:, col].max()
        if col_max > col_min:
            arr[:, col] = (arr[:, col] - col_min) / (col_max - col_min)
        else:
            arr[:, col] = 0.0

    return arr


def run_autoencoder(spans: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Trains a small PyTorch MLP autoencoder on the structured feature matrix.
    Returns:
      recon_errors  float[N]  MSE reconstruction error per span
      ae_labels     int[N]    -1 anomaly, 1 normal (threshold at P{TRIAGE_AE_THRESHOLD_PCT})
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("ERROR: pip install torch")

    feat_matrix = _build_feature_matrix(spans)
    n_features  = feat_matrix.shape[1]
    n           = len(spans)

    print(f"  AE: training autoencoder on {n} spans x {n_features} features ...")

    class Autoencoder(nn.Module):
        def __init__(self, in_dim: int, hidden: int, bottleneck: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim,    hidden),     nn.ReLU(),
                nn.Linear(hidden,    hidden * 2), nn.ReLU(),
                nn.Linear(hidden*2,  bottleneck),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, hidden * 2), nn.ReLU(),
                nn.Linear(hidden * 2, hidden),     nn.ReLU(),
                nn.Linear(hidden,     in_dim),     nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = Autoencoder(n_features, TRIAGE_AE_HIDDEN_DIM, TRIAGE_AE_BOTTLENECK).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=TRIAGE_AE_LR)
    loss_fn = nn.MSELoss(reduction="none")

    X = torch.tensor(feat_matrix, dtype=torch.float32, device=device)

    model.train()
    for epoch in range(TRIAGE_AE_EPOCHS):
        opt.zero_grad()
        out  = model(X)
        loss = loss_fn(out, X).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1}/{TRIAGE_AE_EPOCHS}  loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        recon    = model(X)
        per_span = loss_fn(recon, X).mean(dim=1).cpu().numpy()

    threshold = float(np.percentile(per_span, TRIAGE_AE_THRESHOLD_PCT))
    ae_labels = np.where(per_span > threshold, -1, 1).astype(int)
    n_anom    = int((ae_labels == -1).sum())
    print(f"  AE: threshold={threshold:.6f} (P{TRIAGE_AE_THRESHOLD_PCT}), {n_anom} anomalies flagged")

    return per_span.astype(float), ae_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    for path in (SPAN_FILE, EMB_FILE):
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Run embed step first.")

    print("Loading spans and embeddings ...")
    spans: list[dict] = [
        json.loads(l)
        for l in SPAN_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    embeddings: np.ndarray = np.load(EMB_FILE)

    if len(spans) != len(embeddings):
        sys.exit(
            f"ERROR: spans ({len(spans)}) and embeddings ({len(embeddings)}) are misaligned. "
            "Re-run embed.py."
        )

    n = len(spans)
    print(f"Loaded {n} spans, embeddings shape {embeddings.shape}")

    raw_idx = _build_raw_index()

    # Phase 1: Hard rules
    print("\n[Phase 1] Applying hard rules ...")
    valid_durs = [s.get("duration_ms") or 0.0 for s in spans if (s.get("duration_ms") or 0) > 0]
    p95_ms     = float(np.percentile(valid_durs, TRIAGE_LATENCY_PERCENTILE)) if valid_durs else 0.0
    print(f"  P{TRIAGE_LATENCY_PERCENTILE} latency threshold: {p95_ms:.1f} ms")

    rule_fired_list: list[bool]      = []
    rule_types_list: list[list[str]] = []
    for s in spans:
        raw = raw_idx.get(s["span_id"], {})
        fired, ptypes = apply_rules(s, raw, p95_ms)
        rule_fired_list.append(fired)
        rule_types_list.append(ptypes)

    n_rules = sum(rule_fired_list)
    print(f"  Hard rules fired on {n_rules}/{n} spans")

    # Phase 2a: LOF
    print("\n[Phase 2a] Running LOF ...")
    lof_scores, lof_labels = run_lof(embeddings)

    # Phase 2b: Autoencoder
    print("\n[Phase 2b] Running Autoencoder ...")
    ae_errors, ae_labels = run_autoencoder(spans)

    # Merge
    print("\nMerging results ...")

    def _combined_types(rule_types: list[str], rule_fired: bool, ml_fired: bool) -> list[str]:
        types = list(rule_types)
        if ml_fired:
            if not rule_fired:
                types.append("ml_anomaly")
            else:
                types.append("ml_confirmed")
        return types

    scores: list[dict[str, Any]] = []
    for i, s in enumerate(spans):
        rule_fired  = rule_fired_list[i]
        rule_types  = rule_types_list[i]
        is_anom_lof = bool(lof_labels[i] == -1)
        is_anom_ae  = bool(ae_labels[i] == -1)

        scores.append({
            "span_id":               s["span_id"],
            "rule_fired":            rule_fired,
            "rule_types":            rule_types,
            "lof_score":             round(float(lof_scores[i]), 6),
            "is_anomaly_lof":        is_anom_lof,
            "is_problematic_lof":    rule_fired or is_anom_lof,
            "problem_types_lof":     _combined_types(rule_types, rule_fired, is_anom_lof),
            "ae_reconstruction_error": round(float(ae_errors[i]), 8),
            "is_anomaly_ae":         is_anom_ae,
            "is_problematic_ae":     rule_fired or is_anom_ae,
            "problem_types_ae":      _combined_types(rule_types, rule_fired, is_anom_ae),
        })

    SCORES_FILE.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"Saved triage scores -> {SCORES_FILE}")

    # Report
    def _type_counts(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in scores:
            for pt in rec[key]:
                counts[pt] = counts.get(pt, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    n_prob_lof = sum(1 for r in scores if r["is_problematic_lof"])
    n_prob_ae  = sum(1 for r in scores if r["is_problematic_ae"])

    report: dict[str, Any] = {
        "total_spans":              n,
        "p95_latency_threshold_ms": round(p95_ms, 2),
        "lof": {
            "problematic":  n_prob_lof,
            "healthy":      n - n_prob_lof,
            "problem_types": _type_counts("problem_types_lof"),
        },
        "ae": {
            "problematic":  n_prob_ae,
            "healthy":      n - n_prob_ae,
            "problem_types": _type_counts("problem_types_ae"),
        },
        "rules": {
            "fired":         n_rules,
            "problem_types": _type_counts("rule_types"),
        },
    }

    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved triage report -> {REPORT_FILE}")
    print(f"\nSummary:")
    print(f"  LOF  : {n_prob_lof} problematic, {n - n_prob_lof} healthy")
    print(f"  AE   : {n_prob_ae} problematic,  {n - n_prob_ae} healthy")
    print(f"  Rules: {n_rules} fired")


if __name__ == "__main__":
    run()

main = run
