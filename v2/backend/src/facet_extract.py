"""
Step 4 -- LLM-first facet extraction for error traces.

For each error trace, calls mistral-large-latest to extract a structured
issue facet describing the failure. Then embeds the facet "summary" field
with mistral-embed for downstream cosine-similarity clustering.

Reads:   data/error_traces.jsonl
Writes:  data/trace_facets.jsonl        (one facet record per error trace)
         data/facet_embeddings.npy      (float32 [N_errors, dim])

Facet schema per trace:
{
    "trace_id":           str,
    "issue_type":         str,   # 3-5 word failure category
    "root_cause":         str,   # one sentence
    "affected_component": str,   # "model" | "tool" | "retrieval" | "api" | "unknown"
    "summary":            str,   # one sentence combining issue_type + root_cause
    "heuristic_fallback": bool,  # True if LLM call failed and we used heuristics
}

Usage:
    python -m src.facet_extract
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
from pathlib import Path

from mistralai.client import Mistral

from src.config import (
    MISTRAL_API_KEY,
    FACET_MODEL,
    EMBED_MODEL,
    FACET_BATCH_SIZE,
    FACET_MAX_RETRIES,
    MISTRAL_CLIENT_KWARGS,
)

ERROR_FILE    = Path("data/error_traces.jsonl")
FACETS_FILE   = Path("data/trace_facets.jsonl")
FACET_EMB_FILE = Path("data/facet_embeddings.npy")


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_facet(trace: dict) -> dict:
    """Build a structured facet from deterministic fields when LLM is unavailable."""
    err_msgs    = trace.get("error_messages") or []
    exc_types   = trace.get("exception_types") or []
    http_errors = trace.get("http_errors") or []
    tools       = trace.get("tools_called") or []
    rule_tags   = trace.get("rule_tags") or []
    ops         = trace.get("operations") or ["operation"]

    top_op = ops[0] if ops else "operation"

    if exc_types and tools:
        issue_type = f"Tool Exception: {tools[0]}"
        component  = "tool"
        root_cause = f"Exception {exc_types[0]} raised during {tools[0]} execution."
    elif exc_types:
        issue_type = f"Exception in {top_op.title()}"
        component  = "model"
        root_cause = f"Exception {exc_types[0]} raised in {top_op} span."
    elif http_errors:
        issue_type = f"HTTP {http_errors[0]} Error"
        component  = "api"
        root_cause = f"Received HTTP {http_errors[0]} from downstream service."
    elif err_msgs:
        short = err_msgs[0][:60]
        issue_type = f"API Error in {top_op.title()}"
        component  = "api"
        root_cause = f"Error: {short}"
    elif "truncated_output" in rule_tags:
        issue_type = f"Truncated Output"
        component  = "model"
        root_cause = f"Output was cut short due to token limit (finish_reason=length)."
    elif "high_latency" in rule_tags:
        issue_type = f"High Latency"
        component  = "model"
        root_cause = f"Trace duration exceeded the P95 latency threshold."
    elif "empty_response" in rule_tags:
        issue_type = f"Empty Response"
        component  = "model"
        root_cause = f"Zero output tokens returned on a generative span."
    else:
        issue_type = f"Anomalous Trace"
        component  = "unknown"
        root_cause = "Trace flagged as anomalous by the autoencoder detector."

    summary = f"{issue_type}: {root_cause}"
    return {
        "trace_id":           trace["trace_id"],
        "issue_type":         issue_type,
        "root_cause":         root_cause,
        "affected_component": component,
        "summary":            summary,
        "heuristic_fallback": True,
    }


# ---------------------------------------------------------------------------
# LLM facet extraction
# ---------------------------------------------------------------------------

def _build_prompt(trace: dict) -> str:
    ops    = ", ".join(trace.get("operations") or ["-"])
    models = ", ".join(trace.get("models") or ["-"])
    tools  = ", ".join(trace.get("tools_called") or ["-"])
    errs   = "; ".join((trace.get("error_messages") or [])[:3]) or "-"
    excs   = "; ".join((trace.get("exception_types") or [])[:3]) or "-"
    http   = ", ".join(str(x) for x in (trace.get("http_errors") or [])) or "-"
    in_t   = trace.get("input_tokens") or 0
    out_t  = trace.get("output_tokens") or 0
    dur    = trace.get("total_duration_ms") or 0
    spans  = trace.get("span_count") or 1
    finish = ", ".join(trace.get("finish_reasons") or []) or "-"

    return (
        "You are an expert LLM observability engineer.\n"
        "Analyze this error trace from a production AI pipeline and extract the failure facet.\n\n"
        "TRACE SUMMARY:\n"
        f"- Spans: {spans}, Total duration: {dur:.0f}ms\n"
        f"- Operations: {ops}\n"
        f"- Models: {models}\n"
        f"- Tools called: {tools}\n"
        f"- Error messages: {errs}\n"
        f"- Exception types: {excs}\n"
        f"- HTTP error codes: {http}\n"
        f"- Finish reasons: {finish}\n"
        f"- Input tokens: {in_t}, Output tokens: {out_t}\n\n"
        "Respond as JSON only — no markdown, no explanation:\n"
        "{\n"
        '  "issue_type": "<3-5 word failure category, e.g. Tool Connection Timeout>",\n'
        '  "root_cause": "<one sentence root cause>",\n'
        '  "affected_component": "<one of: model | tool | retrieval | api | unknown>",\n'
        '  "summary": "<one concise sentence combining issue_type and root_cause>"\n'
        "}"
    )


def _extract_facet_llm(client: Mistral, trace: dict) -> dict | None:
    prompt = _build_prompt(trace)
    for attempt in range(FACET_MAX_RETRIES):
        try:
            resp = client.chat.complete(
                model=FACET_MODEL,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                parts = content.split("```")
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content.strip())
            if all(k in data for k in ("issue_type", "root_cause", "affected_component", "summary")):
                return {
                    "trace_id":           trace["trace_id"],
                    "issue_type":         str(data["issue_type"]),
                    "root_cause":         str(data["root_cause"]),
                    "affected_component": str(data["affected_component"]),
                    "summary":            str(data["summary"]),
                    "heuristic_fallback": False,
                }
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    WARNING: LLM facet attempt {attempt+1} failed ({exc}). Retrying in {wait}s ...")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_batch(client: Mistral, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, inputs=texts)
    return [e.embedding for e in resp.data]


def embed_facets(facets: list[dict]) -> np.ndarray:
    summaries = [f["summary"] for f in facets]
    batch_size = 64
    vectors: list[list[float]] = []

    with Mistral(api_key=MISTRAL_API_KEY, **MISTRAL_CLIENT_KWARGS) as client:
        for i in range(0, len(summaries), batch_size):
            chunk = summaries[i : i + batch_size]
            vectors.extend(_embed_batch(client, chunk))
            print(f"  embedded facets {min(i + batch_size, len(summaries))}/{len(summaries)}")

    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    if not ERROR_FILE.exists() or ERROR_FILE.stat().st_size == 0:
        sys.exit(f"ERROR: {ERROR_FILE} not found or empty. Run triage step first.")

    error_traces = [
        json.loads(l)
        for l in ERROR_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    if not error_traces:
        print("No error traces to process.")
        FACETS_FILE.write_text("")
        np.save(FACET_EMB_FILE, np.empty((0, 1024), dtype=np.float32))
        return

    print(f"Extracting facets for {len(error_traces)} error traces ...")

    facets: list[dict] = []

    if not MISTRAL_API_KEY:
        print("WARNING: No MISTRAL_API_KEY — using heuristic fallback for all traces.")
        for t in error_traces:
            facets.append(_heuristic_facet(t))
    else:
        with Mistral(api_key=MISTRAL_API_KEY, **MISTRAL_CLIENT_KWARGS) as client:
            for i, trace in enumerate(error_traces):
                print(f"  [{i+1}/{len(error_traces)}] trace_id={trace['trace_id'][:16]}...")
                facet = _extract_facet_llm(client, trace)
                if facet is None:
                    print(f"    -> LLM failed, using heuristic fallback")
                    facet = _heuristic_facet(trace)
                facets.append(facet)
                # Small delay to respect rate limits
                if (i + 1) % FACET_BATCH_SIZE == 0 and i + 1 < len(error_traces):
                    time.sleep(1)

    # Save facets
    with open(FACETS_FILE, "w", encoding="utf-8") as f:
        for facet in facets:
            f.write(json.dumps(facet) + "\n")
    print(f"Saved {len(facets)} facets -> {FACETS_FILE}")

    # Embed facet summaries
    print(f"\nEmbedding {len(facets)} facet summaries with {EMBED_MODEL} ...")
    arr = embed_facets(facets)
    np.save(FACET_EMB_FILE, arr)
    print(f"Saved facet embeddings {arr.shape} -> {FACET_EMB_FILE}")

    heuristic_count = sum(1 for f in facets if f.get("heuristic_fallback"))
    print(f"\nFacet extraction complete: {len(facets)} facets "
          f"({heuristic_count} heuristic fallbacks)")


if __name__ == "__main__":
    run()

main = run
