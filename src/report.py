"""
Step 6 -- Generate a human-readable cluster report.

Prints a two-level tree to stdout and saves it to data/report.md.

Reads:   data/macro_clusters.json
         data/micro_clusters.json
         data/spans.jsonl
Writes:  data/report.md

Usage:
    python -m src.report
    python -m src.report --examples 3   # show N span examples per cluster
"""

import json
import argparse
import sys
import io
from pathlib import Path

# Force UTF-8 on Windows so LLM-generated labels print cleanly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MACRO_FILE = Path("data/macro_clusters.json")
MICRO_FILE = Path("data/micro_clusters.json")
SPAN_FILE      = Path("data/spans.jsonl")
PROB_SPAN_FILE = Path("data/problematic_spans.jsonl")
OUT            = Path("data/report.md")


def _format_span(span: dict) -> str:
    dur = f"{span['duration_ms']:.0f}ms" if span.get("duration_ms") else "?"
    clean_types = [t for t in span.get("problem_types", []) if t not in ("ml_anomaly", "ml_confirmed")]
    ptypes = ", ".join(clean_types)
    ptag = f"  tags=[{ptypes}]" if ptypes else ""
    return (
        f"    span_id={span['span_id'][:16]}...  "
        f"op={span.get('operation_name','?')}  "
        f"model={span.get('model','?')}  "
        f"status={span.get('status','?')}  "
        f"dur={dur}{ptag}\n"
        f"      err: {span.get('error_message','') or '--'}\n"
        f"      in:  {span.get('input_summary','')[:60] or '--'}"
    )


def run(n_examples: int = 0) -> None:
    if not MACRO_FILE.exists() or not MICRO_FILE.exists():
        raise SystemExit(
            "ERROR: cluster files not found. Run cluster_macro, cluster_micro, "
            "and label steps first."
        )

    macro_data = json.loads(MACRO_FILE.read_text(encoding="utf-8"))
    micro_data = json.loads(MICRO_FILE.read_text(encoding="utf-8"))

    all_spans: dict[str, dict] = {}
    if n_examples > 0:
        if SPAN_FILE.exists():
            for l in SPAN_FILE.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    s = json.loads(l)
                    all_spans[s["span_id"]] = s
        if PROB_SPAN_FILE.exists():
            for l in PROB_SPAN_FILE.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    s = json.loads(l)
                    all_spans[s["span_id"]] = s

    # ---- Counts --------------------------------------------------------------
    total_spans  = sum(c["size"] for c in macro_data.values())
    total_noise  = next((c["size"] for c in macro_data.values() if c["is_noise"]), 0)
    n_macro      = sum(1 for c in macro_data.values() if not c["is_noise"])

    lines: list[str] = []
    lines.append("# Cluster Report")
    lines.append("")
    lines.append(f"**Total spans:** {total_spans}  |  "
                 f"**Macro clusters:** {n_macro}  |  "
                 f"**Noise (unclustered):** {total_noise}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sort macro clusters by size descending, noise last
    sorted_macro = sorted(
        macro_data.items(),
        key=lambda x: (x[1]["is_noise"], -x[1]["size"]),
    )

    for macro_id, cluster in sorted_macro:
        noise_tag = " [NOISE]" if cluster["is_noise"] else ""
        lines.append(
            f"## [{macro_id}] {cluster.get('label') or '(unlabelled)'}{noise_tag}"
            f"  --  {cluster['size']} spans"
        )
        if cluster.get("description"):
            lines.append(f"> {cluster['description']}")
        lines.append("")

        # Span examples for macro cluster
        if n_examples > 0 and not cluster["is_noise"]:
            for sid in cluster["span_ids"][:n_examples]:
                span = all_spans.get(sid)
                if span:
                    lines.append(_format_span(span))
            lines.append("")

        # Micro subclusters
        sub = micro_data.get(macro_id, {})
        if sub:
            sorted_sub = sorted(
                sub.items(),
                key=lambda x: (x[1]["is_noise"], -x[1]["size"]),
            )
            for micro_id, mc in sorted_sub:
                noise_tag_m = " [noise]" if mc["is_noise"] else ""
                lines.append(
                    f"  - **[{macro_id}.{micro_id}]** "
                    f"{mc.get('label') or '(unlabelled)'}{noise_tag_m}"
                    f"  --  {mc['size']} spans"
                )
                if mc.get("description"):
                    lines.append(f"    > {mc['description']}")

                if n_examples > 0 and not mc["is_noise"]:
                    for sid in mc["span_ids"][:n_examples]:
                        span = all_spans.get(sid)
                        if span:
                            lines.append(_format_span(span))
            lines.append("")
        else:
            lines.append("  *(no subclusters -- macro cluster too small)*")
            lines.append("")

    report = "\n".join(lines)
    print(report)
    OUT.write_text(report, encoding="utf-8")
    print(f"\nReport saved -> {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print cluster summary report.")
    parser.add_argument(
        "--examples", type=int, default=0,
        help="Number of span examples to show per cluster (default: 0 = none)"
    )
    args = parser.parse_args()
    run(n_examples=args.examples)
