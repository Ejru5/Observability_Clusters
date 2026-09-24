"""
Step 6 -- Generate a human-readable trace cluster report.

Reads:   data/trace_clusters.json
         data/trace_facets.jsonl
         data/error_traces.jsonl
         data/triage_report.json
Writes:  data/report.md

Usage:
    python -m src.report
    python -m src.report --examples 3
"""

from __future__ import annotations

import json
import argparse
import io
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CLUSTER_FILE  = Path("data/trace_clusters.json")
FACETS_FILE   = Path("data/trace_facets.jsonl")
ERROR_FILE    = Path("data/error_traces.jsonl")
TRIAGE_REPORT = Path("data/triage_report.json")
OUT           = Path("data/report.md")


def _format_trace(trace: dict, facet: dict | None, n_examples: int) -> list[str]:
    dur    = f"{trace.get('total_duration_ms', 0):.0f}ms"
    spans  = trace.get("span_count", 1)
    ops    = ", ".join(trace.get("operations") or [])
    models = ", ".join(trace.get("models") or [])
    tools  = ", ".join(trace.get("tools_called") or ["-"])
    flag   = trace.get("flagged_by") or "rules"
    errs   = "; ".join((trace.get("error_messages") or [])[:2]) or "-"

    lines = [
        f"  - `{trace['trace_id'][:20]}...`  "
        f"dur={dur}  spans={spans}  flag={flag}",
        f"    ops: {ops or '-'}  models: {models or '-'}  tools: {tools}",
        f"    error: {errs[:100]}",
    ]
    if facet:
        lines.append(f"    facet: *{facet.get('summary', '')}*")
    return lines


def run(n_examples: int = 2) -> None:
    if not CLUSTER_FILE.exists():
        raise SystemExit(
            "ERROR: trace_clusters.json not found. Run cluster_traces step first."
        )

    cluster_data = json.loads(CLUSTER_FILE.read_text(encoding="utf-8"))
    clusters     = cluster_data.get("clusters", {})
    unique_errors = cluster_data.get("unique_errors", [])

    # Load facets and error traces indexed by trace_id
    facet_by_id: dict[str, dict] = {}
    if FACETS_FILE.exists():
        for l in FACETS_FILE.read_text(encoding="utf-8").splitlines():
            if l.strip():
                f = json.loads(l)
                facet_by_id[f["trace_id"]] = f

    trace_by_id: dict[str, dict] = {}
    if ERROR_FILE.exists():
        for l in ERROR_FILE.read_text(encoding="utf-8").splitlines():
            if l.strip():
                t = json.loads(l)
                trace_by_id[t["trace_id"]] = t

    # Triage stats header
    triage_stats: dict = {}
    if TRIAGE_REPORT.exists():
        triage_stats = json.loads(TRIAGE_REPORT.read_text(encoding="utf-8"))

    total     = triage_stats.get("total_traces", "?")
    n_errors  = triage_stats.get("error_traces", len(trace_by_id))
    n_healthy = triage_stats.get("healthy_traces", "?")
    err_rate  = triage_stats.get("error_rate_pct", "?")
    by_rules  = triage_stats.get("caught_by_rules", "?")
    by_ae     = triage_stats.get("caught_by_autoencoder", "?")

    lines: list[str] = []
    lines.append("# Trace Cluster Report")
    lines.append("")
    lines.append(
        f"**Total traces:** {total}  |  "
        f"**Error traces:** {n_errors} ({err_rate}%)  |  "
        f"**Healthy traces:** {n_healthy}"
    )
    lines.append(
        f"**Caught by hard rules:** {by_rules}  |  "
        f"**Caught by autoencoder:** {by_ae}"
    )
    lines.append("")
    lines.append(
        f"**Error clusters:** {len(clusters)}  |  "
        f"**Unique errors (singletons):** {len(unique_errors)}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Error Clusters ----
    if clusters:
        lines.append("## Error Clusters")
        lines.append("")

        sorted_clusters = sorted(
            clusters.values(),
            key=lambda c: -c["size"],
        )

        for cluster in sorted_clusters:
            cid   = cluster["cluster_id"]
            label = cluster.get("label") or "(unlabelled)"
            desc  = cluster.get("description") or ""
            size  = cluster["size"]
            comps = ", ".join(cluster.get("affected_components") or [])
            dom   = cluster.get("dominant_issue_type") or ""

            lines.append(f"### [{cid}] {label}  —  {size} traces")
            if desc:
                lines.append(f"> {desc}")
            if dom or comps:
                lines.append(f"> **Issue:** {dom}  |  **Component:** {comps}")
            lines.append("")

            if n_examples > 0:
                for tid in cluster.get("trace_ids", [])[:n_examples]:
                    trace  = trace_by_id.get(tid)
                    facet  = facet_by_id.get(tid)
                    if trace:
                        lines.extend(_format_trace(trace, facet, n_examples))
                lines.append("")

    # ---- Unique Errors ----
    if unique_errors:
        lines.append("---")
        lines.append("")
        lines.append("## Unique Errors")
        lines.append("")
        lines.append(
            "> These traces had no similar peers and could not be grouped into a cluster. "
            "Each represents a distinct one-off failure."
        )
        lines.append("")

        for ue in unique_errors:
            tid    = ue.get("trace_id", "")
            issue  = ue.get("issue_type", "Unknown")
            summary = ue.get("summary", "")
            comp   = ue.get("component", "unknown")
            trace  = trace_by_id.get(tid)
            dur    = f"{trace.get('total_duration_ms', 0):.0f}ms" if trace else "?"
            flag   = trace.get("flagged_by", "?") if trace else "?"

            lines.append(
                f"- `{tid[:20]}...`  **{issue}** [{comp}]  dur={dur}  flag={flag}"
            )
            if summary:
                lines.append(f"  > {summary}")

        lines.append("")

    report = "\n".join(lines)
    print(report)
    OUT.write_text(report, encoding="utf-8")
    print(f"\nReport saved -> {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print trace cluster report.")
    parser.add_argument("--examples", type=int, default=2,
                        help="Trace examples to show per cluster (default: 2)")
    args = parser.parse_args()
    run(n_examples=args.examples)

main = run
