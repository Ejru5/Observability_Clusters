"""
Step 3 -- Triage: bifurcate traces into error vs healthy.

Two-phase, high-recall design — never miss an error trace:

  Phase 1 — Hard rules (deterministic, always fire first):
    - has_error == True  (any span had status=error)
    - exception_types non-empty
    - error_messages non-empty
    - http_errors (HTTP >= 400) non-empty
    - finish_reasons contains "length"  (truncated output)
    - output_tokens == 0 on a trace that has chat/completion spans
    - total_duration_ms > P95 of all traces  (high latency)

  Phase 2 — Autoencoder on traces NOT caught by Phase 1:
    Trained on clean (Phase-1-passed) traces only (semi-supervised).
    Threshold at P75 reconstruction error = high recall, catches subtle errors.

    Feature matrix (10 columns per trace):
      0  total_duration_ms      (normalised)
      1  span_count             (normalised)
      2  input_tokens           (normalised)
      3  output_tokens          (normalised)
      4  output_token_ratio     output / (input + output + 1)
      5  has_chat_span          bool
      6  has_tool_span          bool
      7  tool_diversity         len(unique tools) / span_count
      8  model_diversity        len(unique models) / span_count
      9  embedding_distance     cosine distance from mean embedding (semantic signal)

Reads:   data/traces.jsonl
         data/trace_embeddings.npy
Writes:  data/error_traces.jsonl
         data/healthy_traces.jsonl
         data/triage_report.json

Usage:
    python -m src.triage
"""

from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from typing import Any

from src.config import (
    TRIAGE_LATENCY_PERCENTILE,
    TRIAGE_AE_EPOCHS,
    TRIAGE_AE_LR,
    TRIAGE_AE_HIDDEN_DIM,
    TRIAGE_AE_BOTTLENECK,
    TRIAGE_AE_THRESHOLD_PCT,
)

TRACE_FILE   = Path("data/traces.jsonl")
EMB_FILE     = Path("data/trace_embeddings.npy")
ERROR_FILE   = Path("data/error_traces.jsonl")
HEALTHY_FILE = Path("data/healthy_traces.jsonl")
REPORT_FILE  = Path("data/triage_report.json")


# ---------------------------------------------------------------------------
# Phase 1 — Hard rules
# ---------------------------------------------------------------------------

def apply_rules(trace: dict, p95_ms: float) -> tuple[bool, list[str]]:
    """
    Returns (fired, list_of_rule_tags).
    A single fired rule is enough to classify this trace as error.
    """
    tags: list[str] = []

    if trace.get("has_error"):
        tags.append("explicit_error")

    if trace.get("exception_types"):
        tags.append("exception")

    if trace.get("error_messages"):
        if "explicit_error" not in tags:
            tags.append("api_error")

    if trace.get("http_errors"):
        if "api_error" not in tags and "explicit_error" not in tags:
            tags.append("http_error")

    if "length" in (trace.get("finish_reasons") or []):
        tags.append("truncated_output")

    # Empty response: no output tokens but has chat/completion operation
    ops = [o.lower() for o in (trace.get("operations") or [])]
    has_gen = any(o in ("chat", "completion", "fim") for o in ops)
    if has_gen and (trace.get("output_tokens") or 0) == 0:
        tags.append("empty_response")

    # High latency
    dur = trace.get("total_duration_ms") or 0.0
    if p95_ms > 0 and dur > p95_ms:
        tags.append("high_latency")

    return bool(tags), tags


# ---------------------------------------------------------------------------
# Phase 2 — Autoencoder
# ---------------------------------------------------------------------------

def _build_feature_matrix(
    traces: list[dict],
    embeddings: np.ndarray,
    mean_emb: np.ndarray,
) -> np.ndarray:
    """
    Build 10-column numeric feature matrix for AE.
    Column 9 uses cosine distance from overall mean embedding as semantic signal.
    """
    rows = []
    for i, t in enumerate(traces):
        dur      = float(t.get("total_duration_ms") or 0.0)
        n_spans  = float(t.get("span_count") or 1)
        in_tok   = float(t.get("input_tokens") or 0.0)
        out_tok  = float(t.get("output_tokens") or 0.0)
        ops      = [o.lower() for o in (t.get("operations") or [])]
        tools    = t.get("tools_called") or []
        models   = t.get("models") or []

        # Semantic distance from mean (normalised)
        emb = embeddings[i]
        norm_emb  = emb  / (np.linalg.norm(emb)  + 1e-8)
        norm_mean = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        cos_sim   = float(np.dot(norm_emb, norm_mean))
        cos_dist  = 1.0 - cos_sim   # 0 = identical, 2 = opposite

        rows.append([
            dur,
            n_spans,
            in_tok,
            out_tok,
            out_tok / (in_tok + out_tok + 1.0),
            1.0 if any(o in ("chat", "completion", "fim") for o in ops) else 0.0,
            1.0 if tools else 0.0,
            len(tools) / n_spans,
            len(models) / n_spans,
            cos_dist,
        ])

    arr = np.array(rows, dtype=np.float32)

    # Min-max normalise continuous cols (0-4, 9); skip binary (5-8)
    for col in list(range(5)) + [9]:
        col_min, col_max = arr[:, col].min(), arr[:, col].max()
        if col_max > col_min:
            arr[:, col] = (arr[:, col] - col_min) / (col_max - col_min)
        else:
            arr[:, col] = 0.0

    return arr


def run_autoencoder(
    traces: list[dict],
    embeddings: np.ndarray,
    mean_emb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train AE on these traces (assumed clean / phase-1-passed).
    Returns (recon_errors [N], ae_labels [N])  where -1 = anomaly, 1 = normal.
    Threshold at TRIAGE_AE_THRESHOLD_PCT (default 75) for high recall.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("ERROR: pip install torch")

    feat = _build_feature_matrix(traces, embeddings, mean_emb)
    n, n_features = feat.shape

    print(f"  AE: training on {n} clean traces x {n_features} features ...")

    class Autoencoder(nn.Module):
        def __init__(self, in_dim: int, hidden: int, bottleneck: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim,   hidden),      nn.ReLU(),
                nn.Linear(hidden,   hidden * 2),  nn.ReLU(),
                nn.Linear(hidden*2, bottleneck),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, hidden * 2), nn.ReLU(),
                nn.Linear(hidden * 2, hidden),     nn.ReLU(),
                nn.Linear(hidden,     in_dim),     nn.Sigmoid(),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = Autoencoder(n_features, TRIAGE_AE_HIDDEN_DIM, TRIAGE_AE_BOTTLENECK).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=TRIAGE_AE_LR)
    loss_fn = nn.MSELoss(reduction="none")
    X       = torch.tensor(feat, dtype=torch.float32, device=device)

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
        per_trace = loss_fn(recon, X).mean(dim=1).cpu().numpy()

    threshold = float(np.percentile(per_trace, TRIAGE_AE_THRESHOLD_PCT))
    ae_labels = np.where(per_trace > threshold, -1, 1).astype(int)
    n_anom    = int((ae_labels == -1).sum())
    print(f"  AE: threshold={threshold:.6f} (P{TRIAGE_AE_THRESHOLD_PCT}), {n_anom} anomalies flagged")

    return per_trace.astype(float), ae_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    for path in (TRACE_FILE, EMB_FILE):
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Run embed step first.")

    print("Loading traces and embeddings ...")
    traces: list[dict] = [
        json.loads(l)
        for l in TRACE_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    embeddings: np.ndarray = np.load(EMB_FILE)

    if len(traces) != len(embeddings):
        sys.exit(
            f"ERROR: traces ({len(traces)}) and embeddings ({len(embeddings)}) misaligned. "
            "Re-run embed step."
        )

    n = len(traces)
    print(f"Loaded {n} traces, embeddings shape {embeddings.shape}")

    # ---- Phase 1: Hard rules -------------------------------------------
    print(f"\n[Phase 1] Applying hard rules ...")
    valid_durs = [t.get("total_duration_ms") or 0.0 for t in traces if (t.get("total_duration_ms") or 0) > 0]
    p95_ms = float(np.percentile(valid_durs, TRIAGE_LATENCY_PERCENTILE)) if valid_durs else 0.0
    print(f"  P{TRIAGE_LATENCY_PERCENTILE} latency threshold: {p95_ms:.1f} ms")

    rule_results: list[tuple[bool, list[str]]] = []
    for t in traces:
        fired, tags = apply_rules(t, p95_ms)
        rule_results.append((fired, tags))

    error_by_rule    = [i for i, (fired, _) in enumerate(rule_results) if fired]
    clean_after_rules = [i for i, (fired, _) in enumerate(rule_results) if not fired]
    print(f"  Hard rules: {len(error_by_rule)} error traces, {len(clean_after_rules)} passed to AE")

    # ---- Phase 2: Autoencoder on clean traces --------------------------
    ae_error_indices: list[int] = []
    ae_recon_errors_map: dict[int, float] = {}

    if len(clean_after_rules) >= 5:
        print(f"\n[Phase 2] Running Autoencoder on {len(clean_after_rules)} clean traces ...")
        clean_traces = [traces[i] for i in clean_after_rules]
        clean_embs   = embeddings[clean_after_rules]
        mean_emb     = embeddings.mean(axis=0)   # mean of ALL traces as reference

        recon_errors, ae_labels = run_autoencoder(clean_traces, clean_embs, mean_emb)

        for j, orig_i in enumerate(clean_after_rules):
            ae_recon_errors_map[orig_i] = float(recon_errors[j])
            if ae_labels[j] == -1:
                ae_error_indices.append(orig_i)

        print(f"  AE flagged {len(ae_error_indices)} additional error traces")
    else:
        print(f"\n[Phase 2] Skipping AE (only {len(clean_after_rules)} clean traces, need >= 5)")

    # ---- Merge results -------------------------------------------------
    all_error_indices = sorted(set(error_by_rule) | set(ae_error_indices))
    error_set = set(all_error_indices)

    error_traces:   list[dict] = []
    healthy_traces: list[dict] = []

    for i, t in enumerate(traces):
        rec = dict(t)
        rec["is_error"] = i in error_set
        rec["ae_recon_error"] = ae_recon_errors_map.get(i)

        if i in error_set:
            if i in set(error_by_rule):
                rec["flagged_by"] = "rules"
                rec["rule_tags"]  = rule_results[i][1]
            else:
                rec["flagged_by"] = "autoencoder"
                rec["rule_tags"]  = []
            error_traces.append(rec)
        else:
            rec["flagged_by"] = None
            rec["rule_tags"]  = []
            healthy_traces.append(rec)

    # Write output files
    with open(ERROR_FILE, "w", encoding="utf-8") as f:
        for t in error_traces:
            f.write(json.dumps(t) + "\n")
    print(f"\nSaved {len(error_traces)} error traces -> {ERROR_FILE}")

    with open(HEALTHY_FILE, "w", encoding="utf-8") as f:
        for t in healthy_traces:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(healthy_traces)} healthy traces -> {HEALTHY_FILE}")

    # Triage report
    rule_tag_counts: dict[str, int] = {}
    for fired, tags in rule_results:
        if fired:
            for tag in tags:
                rule_tag_counts[tag] = rule_tag_counts.get(tag, 0) + 1
    rule_tag_counts = dict(sorted(rule_tag_counts.items(), key=lambda x: -x[1]))

    report: dict[str, Any] = {
        "total_traces":              n,
        "error_traces":              len(error_traces),
        "healthy_traces":            len(healthy_traces),
        "error_rate_pct":            round(len(error_traces) / n * 100, 1) if n else 0,
        "p95_latency_threshold_ms":  round(p95_ms, 2),
        "caught_by_rules":           len(error_by_rule),
        "caught_by_autoencoder":     len(ae_error_indices),
        "rule_tag_breakdown":        rule_tag_counts,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved triage report -> {REPORT_FILE}")

    print(f"\nSummary:")
    print(f"  Total : {n} traces")
    print(f"  Error : {len(error_traces)} ({report['error_rate_pct']}%)")
    print(f"    -> Rules: {len(error_by_rule)}  |  AE: {len(ae_error_indices)}")
    print(f"  Healthy: {len(healthy_traces)}")


if __name__ == "__main__":
    run()

main = run
