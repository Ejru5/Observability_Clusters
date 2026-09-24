"""
Step 2a -- Normalise raw spans, then roll them up into traces.

Reads:   data/raw_spans.jsonl
Writes:  data/spans.jsonl        (normalised spans, intermediate)
         data/traces.jsonl       (one record per trace_id)

Each trace record:
{
    "trace_id":          str,
    "span_ids":          [str, ...],
    "span_count":        int,
    "start_time":        str,
    "end_time":          str,
    "total_duration_ms": float,
    "models":            [str, ...],
    "operations":        [str, ...],
    "tools_called":      [str, ...],
    "has_error":         bool,
    "error_messages":    [str, ...],
    "exception_types":   [str, ...],
    "finish_reasons":    [str, ...],
    "input_tokens":      int,
    "output_tokens":     int,
    "http_errors":       [int, ...],   # HTTP status codes >= 400
    "trace_text":        str,          # concatenated summary used for embedding
}

Usage:
    python -m src.rollup_traces
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

RAW_FILE   = Path("data/raw_spans.jsonl")
SPAN_FILE  = Path("data/spans.jsonl")
TRACE_FILE = Path("data/traces.jsonl")


# ---------------------------------------------------------------------------
# Span normalisation  (same logic as v1 embed.py)
# ---------------------------------------------------------------------------

def _parse_ms(raw: dict) -> float | None:
    s, e = raw.get("start_time"), raw.get("end_time")
    if not (s and e):
        return None
    try:
        return (
            datetime.fromisoformat(str(e)) - datetime.fromisoformat(str(s))
        ).total_seconds() * 1000
    except Exception:
        return None


def normalize(raw: dict) -> dict:
    attrs: dict = raw.get("span_attributes") or {}
    span_id = raw.get("span_id") or ""

    input_text  = (raw.get("input_messages")
                   or attrs.get("gen_ai.input.messages") or "")
    output_text = (raw.get("output_messages")
                   or attrs.get("gen_ai.output.messages") or "")

    model = (raw.get("response_model")
             or raw.get("request_model")
             or attrs.get("gen_ai.response.model")
             or attrs.get("gen_ai.request.model") or "")

    status = (raw.get("status_code") or "Unset").lower()

    dur_ns = raw.get("duration_ns")
    duration_ms: float | None = dur_ns / 1_000_000 if dur_ns else _parse_ms(raw)

    # HTTP status from span attributes
    http_code_str = str(
        attrs.get("http.response.status_code")
        or attrs.get("http.status_code") or ""
    )
    http_status = int(http_code_str) if http_code_str.isdigit() else None

    finish_reasons = raw.get("response_finish_reasons") or []

    return {
        "span_id":        span_id,
        "trace_id":       raw.get("trace_id", ""),
        "operation_name": (raw.get("operation_name") or raw.get("span_name") or ""),
        "model":          model,
        "tool_called":    (raw.get("tool_name") or attrs.get("gen_ai.tool.name") or ""),
        "duration_ms":    duration_ms,
        "status":         status,
        "error_message":  (raw.get("status_message")
                           or raw.get("error_type")
                           or attrs.get("exception.message") or ""),
        "exception_type": (raw.get("error_type")
                           or attrs.get("exception.type") or ""),
        "input_summary":  str(input_text)[:200],
        "output_summary": str(output_text)[:200],
        "input_tokens":   (raw.get("usage_input_tokens")
                           or attrs.get("gen_ai.usage.input_tokens")),
        "output_tokens":  (raw.get("usage_output_tokens")
                           or attrs.get("gen_ai.usage.output_tokens")),
        "timestamp":      raw.get("start_time", ""),
        "http_status":    http_status,
        "finish_reasons": finish_reasons,
    }


# ---------------------------------------------------------------------------
# Trace rollup
# ---------------------------------------------------------------------------

def _unique(lst: list) -> list:
    seen = set()
    out  = []
    for x in lst:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_trace_text(spans: list[dict]) -> str:
    """
    Concatenate all span key fields into a single string for embedding.
    Format: "[op=chat model=mistral-large status=ok dur=1200ms err=- tool=-]"
    """
    parts = []
    for s in spans:
        dur  = f"{s['duration_ms']:.0f}ms" if s.get("duration_ms") else "?"
        err  = (s.get("error_message") or "-")[:80]
        exc  = (s.get("exception_type") or "-")
        tool = (s.get("tool_called") or "-")
        parts.append(
            f"[op={s.get('operation_name') or '?'} "
            f"model={s.get('model') or '?'} "
            f"status={s.get('status') or '?'} "
            f"dur={dur} "
            f"tool={tool} "
            f"err={err} "
            f"exc={exc}]"
        )
    return " ".join(parts)


def rollup(spans: list[dict]) -> list[dict]:
    """Group normalised spans by trace_id and produce one trace record each."""
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        tid = s.get("trace_id") or s["span_id"]   # fallback: treat span as its own trace
        groups[tid].append(s)

    traces = []
    for tid, slist in groups.items():
        # Sort spans by timestamp
        slist.sort(key=lambda x: x.get("timestamp") or "")

        timestamps = [s.get("timestamp") for s in slist if s.get("timestamp")]
        start_time = timestamps[0] if timestamps else ""
        end_time   = timestamps[-1] if timestamps else ""

        durations      = [s["duration_ms"] for s in slist if s.get("duration_ms")]
        total_duration = sum(durations) if durations else 0.0

        error_messages = _unique([s["error_message"] for s in slist if s.get("error_message")])
        exception_types = _unique([s["exception_type"] for s in slist if s.get("exception_type")])
        http_errors    = _unique([s["http_status"] for s in slist
                                  if s.get("http_status") and s["http_status"] >= 400])
        finish_reasons = []
        for s in slist:
            for fr in (s.get("finish_reasons") or []):
                if fr not in finish_reasons:
                    finish_reasons.append(fr)

        has_error = (
            any(s.get("status") == "error" for s in slist)
            or bool(error_messages)
            or bool(exception_types)
            or bool(http_errors)
        )

        in_tokens  = sum(s.get("input_tokens") or 0 for s in slist)
        out_tokens = sum(s.get("output_tokens") or 0 for s in slist)

        trace_text = _build_trace_text(slist)

        traces.append({
            "trace_id":          tid,
            "span_ids":          [s["span_id"] for s in slist],
            "span_count":        len(slist),
            "start_time":        start_time,
            "end_time":          end_time,
            "total_duration_ms": round(total_duration, 2),
            "models":            _unique([s["model"] for s in slist if s.get("model")]),
            "operations":        _unique([s["operation_name"] for s in slist if s.get("operation_name")]),
            "tools_called":      _unique([s["tool_called"] for s in slist if s.get("tool_called")]),
            "has_error":         has_error,
            "error_messages":    error_messages,
            "exception_types":   exception_types,
            "finish_reasons":    finish_reasons,
            "http_errors":       http_errors,
            "input_tokens":      in_tokens,
            "output_tokens":     out_tokens,
            "trace_text":        trace_text,
        })

    # Sort traces by start_time
    traces.sort(key=lambda x: x.get("start_time") or "")
    return traces


def main() -> None:
    if not RAW_FILE.exists() or RAW_FILE.stat().st_size == 0:
        sys.exit(f"ERROR: {RAW_FILE} not found or empty. Run fetch_spans first.")

    raw_lines = [l for l in RAW_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not raw_lines:
        sys.exit(f"ERROR: {RAW_FILE} contains no data.")

    print(f"Normalising {len(raw_lines)} raw spans ...")
    raws  = [json.loads(l) for l in raw_lines]
    spans = [normalize(r) for r in raws]

    # Save normalised spans (intermediate — used by triage for raw field lookup)
    with open(SPAN_FILE, "w", encoding="utf-8") as f:
        for s in spans:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(spans)} normalised spans -> {SPAN_FILE}")

    # Roll up to traces
    print("Rolling up spans into traces ...")
    traces = rollup(spans)
    print(f"  -> {len(traces)} traces from {len(spans)} spans")

    with open(TRACE_FILE, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(traces)} traces -> {TRACE_FILE}")


if __name__ == "__main__":
    main()

run = main
