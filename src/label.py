"""
Step 5 -- Auto-label macro and micro clusters using mistral-large-latest.

Samples up to 8 member spans per cluster, sends them to mistral-large-latest,
and writes the generated name + description back into both JSON files.

Reads & writes:
    data/macro_clusters.json   (adds "label" and "description" fields)
    data/micro_clusters.json   (adds "label" and "description" fields)

Reads (for context):
    data/spans.jsonl

Usage:
    python -m src.label
"""

import json
import sys
from pathlib import Path

from mistralai.client import Mistral

from src.config import MISTRAL_API_KEY, LABEL_MODEL, MISTRAL_CLIENT_KWARGS

SPAN_FILE      = Path("data/spans.jsonl")
PROB_SPAN_FILE = Path("data/problematic_spans.jsonl")
MACRO_FILE     = Path("data/macro_clusters.json")
MICRO_FILE     = Path("data/micro_clusters.json")


# ---------------------------------------------------------------------------
# Labelling helper
# ---------------------------------------------------------------------------

def _sample_examples(span_ids: list[str], all_spans: dict, n: int = 8) -> str:
    """Build a rich example block for the LLM prompt focusing on failure characteristics."""
    samples = [all_spans[sid] for sid in span_ids[:n] if sid in all_spans]
    lines = []
    for s in samples:
        dur = f"{s['duration_ms']:.0f}ms" if s.get("duration_ms") else "?"
        ptypes = [t for t in s.get("problem_types", []) if t not in ("ml_anomaly", "ml_confirmed")]

        # Build clear failure notes
        failure_notes = []
        if "truncated_output" in ptypes:
            failure_notes.append("TRUNCATED_OUTPUT (finish_reason=length, response cut short)")
        if "high_latency" in ptypes:
            failure_notes.append(f"HIGH_LATENCY ({dur} exceeds P95 threshold)")
        if "empty_response" in ptypes:
            failure_notes.append("EMPTY_RESPONSE (0 output tokens returned)")
        if "tool_error" in ptypes or s.get("tool_called"):
            failure_notes.append(f"TOOL_ERROR (tool={s.get('tool_called') or 'unknown'})")
        if "exception_error" in ptypes or s.get("exception_type"):
            failure_notes.append(f"EXCEPTION ({s.get('exception_type')})")
        if "api_error" in ptypes or s.get("status") == "error":
            failure_notes.append("API_ERROR (status=error or HTTP >= 400)")
        if not failure_notes and ptypes:
            failure_notes.append(", ".join(ptypes))
        if not failure_notes:
            failure_notes.append("STRUCTURAL_ANOMALY (anomalous feature/token profile)")

        err_str = s.get("error_message") or s.get("exception_type") or "-"
        in_str = (s.get("input_summary") or "-")[:120].replace("\n", " ")
        out_str = (s.get("output_summary") or "-")[:120].replace("\n", " ")

        line = (
            f"- op={s.get('operation_name') or '?'}  model={s.get('model') or '?'}  status={s.get('status')}  dur={dur}\n"
            f"  FAILURE: {'; '.join(failure_notes)}\n"
            f"  error: {err_str[:100]}\n"
            f"  user_prompt: {in_str}\n"
            f"  model_output: {out_str}"
        )
        lines.append(line)
    return "\n\n".join(lines)


def _heuristic_label(span_ids: list[str], all_spans: dict) -> dict:
    """Smart heuristic fallback when LLM API is unavailable."""
    samples = [all_spans[sid] for sid in span_ids if sid in all_spans]
    if not samples:
        return {"name": "Error Cluster", "description": "Group of problematic spans."}

    ops: dict[str, int] = {}
    errs: dict[str, int] = {}
    ptypes: dict[str, int] = {}
    tools: dict[str, int] = {}

    for s in samples:
        op = s.get("operation_name") or "operation"
        ops[op] = ops.get(op, 0) + 1
        tool = s.get("tool_called")
        if tool:
            tools[tool] = tools.get(tool, 0) + 1
        err = s.get("exception_type") or s.get("error_message")
        if err:
            err_short = err.strip().split("\n")[0][:40]
            errs[err_short] = errs.get(err_short, 0) + 1
        for pt in s.get("problem_types", []):
            if pt not in ("ml_anomaly", "ml_confirmed"):
                ptypes[pt] = ptypes.get(pt, 0) + 1

    top_op = max(ops.items(), key=lambda x: x[1])[0] if ops else "Span"
    top_tool = max(tools.items(), key=lambda x: x[1])[0] if tools else None
    top_err = max(errs.items(), key=lambda x: x[1])[0] if errs else None
    top_pt = max(ptypes.items(), key=lambda x: x[1])[0] if ptypes else None

    # Construct descriptive semantic failure name
    if top_tool and (top_err or "tool_error" in ptypes):
        name = f"Tool Failure: {top_tool}"
        desc = f"Cluster of {len(span_ids)} spans failing during '{top_tool}' execution."
    elif top_err:
        name = f"{top_op.title()} Error: {top_err}"
        desc = f"Cluster of {len(span_ids)} spans failing with {top_err}."
    elif top_pt == "truncated_output":
        name = f"Truncated {top_op.title()} Outputs"
        desc = f"Cluster of {len(span_ids)} spans with output cut off due to token limits."
    elif top_pt == "high_latency":
        name = f"High Latency in {top_op.title()}"
        desc = f"Cluster of {len(span_ids)} spans with response latency exceeding the P95 threshold."
    elif top_pt == "empty_response":
        name = f"Empty Responses in {top_op.title()}"
        desc = f"Cluster of {len(span_ids)} spans returning 0 output tokens."
    elif top_pt == "api_error":
        name = f"API Errors in {top_op.title()}"
        desc = f"Cluster of {len(span_ids)} spans with API or HTTP error status."
    elif top_pt:
        pt_clean = top_pt.replace("_", " ").title()
        name = f"{top_op.title()} Failure ({pt_clean})"
        desc = f"Cluster of {len(span_ids)} problematic spans flagged for {pt_clean}."
    else:
        name = f"Anomalous {top_op.title()} Spans"
        desc = f"Cluster of {len(span_ids)} anomalous spans identified by statistical outlier detection."

    return {
        "name": name,
        "description": desc,
    }


def label_cluster(
    client: Mistral | None,
    span_ids: list[str],
    all_spans: dict,
    level: str,
    parent_label: str = "",
) -> dict:
    """
    Ask mistral-large-latest to name an error cluster or use heuristic fallback.
    Returns {"name": str, "description": str}.
    """
    if not client:
        return _heuristic_label(span_ids, all_spans)

    examples = _sample_examples(span_ids, all_spans)
    parent_ctx = f" (inside macro error cluster: '{parent_label}')" if parent_label else ""

    prompt = (
        f"You are an expert LLM observability engineer analyzing a cluster of {len(span_ids)} problematic/error spans "
        f"from a production AI pipeline ({level} clustering level{parent_ctx}).\n\n"
        f"Representative spans from this error cluster:\n{examples}\n\n"
        "TASK:\n"
        "Identify the common failure mode, error symptom, or root cause across these spans and generate a concise semantic error label.\n\n"
        "STRICT REQUIREMENTS:\n"
        "1. 'name': A concise 3–5 word semantic failure title describing the error or issue (e.g. 'Truncated Compliance Reports', "
        "'JSON Parsing Error in Tool', 'High Latency Agent Timeout', 'Inventory Tool Connection Failure', 'Empty Completion Responses').\n"
        "2. DO NOT use positive or generic words like 'Successful', 'Completed', 'Normal', or 'Chat Completion' without identifying the error.\n"
        "3. If the primary issue is `TRUNCATED_OUTPUT`, focus on what outputs were cut off (e.g., 'Truncated Travel Itinerary Responses').\n"
        "4. 'description': Exactly one sentence explaining the common failure pattern or root cause.\n\n"
        'Respond as valid JSON only with keys "name" and "description":\n{"name": "...", "description": "..."}'
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
        if isinstance(data, dict) and "name" in data and "description" in data:
            return data
        return _heuristic_label(span_ids, all_spans)
    except Exception as exc:
        print(f"    WARNING: LLM labelling fallback ({exc}).")
        return _heuristic_label(span_ids, all_spans)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    for path in (MACRO_FILE, MICRO_FILE):
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Run the previous steps first.")

    all_spans: dict[str, dict] = {}
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

    macro_data = json.loads(MACRO_FILE.read_text(encoding="utf-8"))
    micro_data = json.loads(MICRO_FILE.read_text(encoding="utf-8"))

    client = None
    if MISTRAL_API_KEY:
        try:
            client = Mistral(api_key=MISTRAL_API_KEY, **MISTRAL_CLIENT_KWARGS)
        except Exception as e:
            print(f"Warning: Could not initialize Mistral client ({e}). Using heuristic labeller.")

    # ---- Label macro clusters -------------------------------------------
    print("Labelling macro clusters ...")
    for cid, cluster in macro_data.items():
        if cluster["is_noise"] or not cluster["span_ids"]:
            cluster["label"]       = "[NOISE]"
            cluster["description"] = "Unclustered spans with no clear pattern."
            continue
        meta = label_cluster(client, cluster["span_ids"], all_spans, "macro")
        cluster["label"]       = meta.get("name", "?")
        cluster["description"] = meta.get("description", "")
        print(f"  macro {cid:>4}: {cluster['label']}")

    # ---- Label micro clusters -------------------------------------------
    print("\nLabelling micro clusters ...")
    for macro_id, sub in micro_data.items():
        parent_label = macro_data.get(macro_id, {}).get("label", "")
        for cid, cluster in sub.items():
            if cluster["is_noise"] or not cluster["span_ids"]:
                cluster["label"]       = "[NOISE]"
                cluster["description"] = "Unclustered spans within this macro group."
                continue
            meta = label_cluster(
                client, cluster["span_ids"], all_spans, "micro", parent_label
            )
            cluster["label"]       = meta.get("name", "?")
            cluster["description"] = meta.get("description", "")
            print(f"    micro {macro_id}.{cid:>3}: {cluster['label']}")

    # Write back
    MACRO_FILE.write_text(json.dumps(macro_data, indent=2), encoding="utf-8")
    MICRO_FILE.write_text(json.dumps(micro_data, indent=2), encoding="utf-8")
    print("\nLabels written to macro_clusters.json and micro_clusters.json.")


if __name__ == "__main__":
    run()
