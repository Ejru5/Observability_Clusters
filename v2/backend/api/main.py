"""
FastAPI backend for the v2 Observability Clusters dashboard (trace-level, LLM-first).

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR       = Path("data")
TRACES_FILE    = DATA_DIR / "traces.jsonl"
ERROR_FILE     = DATA_DIR / "error_traces.jsonl"
HEALTHY_FILE   = DATA_DIR / "healthy_traces.jsonl"
FACETS_FILE    = DATA_DIR / "trace_facets.jsonl"
CLUSTERS_FILE  = DATA_DIR / "trace_clusters.json"
TRIAGE_REPORT  = DATA_DIR / "triage_report.json"
REPORT_FILE    = DATA_DIR / "report.md"
RAW_SPANS_FILE = DATA_DIR / "raw_spans.jsonl"
TRACE_EMB_FILE = DATA_DIR / "trace_embeddings.npy"

ROOT_DIR    = Path(__file__).parent.parent          # v2/backend/
# Use the shared root-level .venv (Observability_Clusters/.venv)
_PROJECT_ROOT = ROOT_DIR.parent.parent               # Observability_Clusters/
VENV_PYTHON   = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_BIN    = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

app = FastAPI(
    title="Observability Clusters v2 API",
    description="Trace-level, LLM-first error clustering pipeline.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------
PIPELINE_ORDER = ["fetch", "embed", "triage", "facet", "cluster", "report"]

PIPELINE_MODULES = {
    "fetch":   "src.fetch_spans",
    "embed":   "src.embed",
    "triage":  "src.triage",
    "facet":   "src.facet_extract",
    "cluster": "src.cluster_traces",
    "report":  "src.report",
}

PIPELINE_OUTPUT_FILES = {
    "fetch":   RAW_SPANS_FILE,
    "embed":   TRACE_EMB_FILE,
    "triage":  ERROR_FILE,
    "facet":   FACETS_FILE,
    "cluster": CLUSTERS_FILE,
    "report":  REPORT_FILE,
}

_full_pipeline_state: dict[str, Any] = {
    "is_running":         False,
    "current_step":       None,
    "current_step_index": 0,
    "total_steps":        len(PIPELINE_ORDER),
    "step_statuses":      {step: "pending" for step in PIPELINE_ORDER},
    "error":              None,
    "start_time":         None,
    "end_time":           None,
    "logs":               [],
}
_running_step: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{path.name} not found. Run the pipeline steps first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PipelineStepStatus(BaseModel):
    step:  str
    done:  bool
    file:  str


class FullPipelineStatusResponse(BaseModel):
    is_running:         bool
    current_step:       str | None
    current_step_index: int
    total_steps:        int
    step_statuses:      dict[str, str]
    error:              str | None
    start_time:         float | None
    end_time:           float | None
    duration_seconds:   float | None


class TraceSummary(BaseModel):
    trace_id:          str
    span_count:        int
    total_duration_ms: float
    operations:        list[str]
    models:            list[str]
    tools_called:      list[str]
    is_error:          bool
    flagged_by:        str | None
    rule_tags:         list[str]
    start_time:        str
    input_tokens:      int
    output_tokens:     int


class TraceDetail(TraceSummary):
    span_ids:         list[str]
    error_messages:   list[str]
    exception_types:  list[str]
    http_errors:      list[int]
    finish_reasons:   list[str]
    ae_recon_error:   float | None
    facet:            dict | None


class ClusterSummary(BaseModel):
    cluster_id:              str
    label:                   str | None
    description:             str | None
    is_unique_error:         bool
    size:                    int
    dominant_issue_type:     str
    affected_components:     list[str]
    representative_trace_id: str


class ClusterDetail(ClusterSummary):
    trace_ids: list[str]


class UniqueError(BaseModel):
    trace_id:   str
    issue_type: str
    summary:    str
    component:  str
    root_cause: str


class RunStepRequest(BaseModel):
    step: str


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _facet_index() -> dict[str, dict]:
    return {f["trace_id"]: f for f in _load_jsonl(FACETS_FILE)}


def _error_trace_index() -> dict[str, dict]:
    return {t["trace_id"]: t for t in _load_jsonl(ERROR_FILE)}


def _trace_to_summary(t: dict) -> TraceSummary:
    return TraceSummary(
        trace_id=t.get("trace_id", ""),
        span_count=t.get("span_count", 0),
        total_duration_ms=t.get("total_duration_ms", 0.0),
        operations=t.get("operations", []),
        models=t.get("models", []),
        tools_called=t.get("tools_called", []),
        is_error=t.get("is_error", False),
        flagged_by=t.get("flagged_by"),
        rule_tags=t.get("rule_tags", []),
        start_time=t.get("start_time", ""),
        input_tokens=t.get("input_tokens", 0),
        output_tokens=t.get("output_tokens", 0),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Observability Clusters v2 API", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=list[PipelineStepStatus], tags=["Pipeline"])
def get_pipeline_status():
    result = []
    for step, path in PIPELINE_OUTPUT_FILES.items():
        done = path.exists() and path.stat().st_size > 0
        result.append(PipelineStepStatus(step=step, done=done, file=str(path)))
    return result


@app.get("/api/pipeline/full-status", response_model=FullPipelineStatusResponse, tags=["Pipeline"])
def get_full_pipeline_status():
    start = _full_pipeline_state.get("start_time")
    end   = _full_pipeline_state.get("end_time") or (
        time.time() if _full_pipeline_state.get("is_running") else None
    )
    duration = (end - start) if (start and end) else None
    return FullPipelineStatusResponse(
        is_running=_full_pipeline_state["is_running"],
        current_step=_full_pipeline_state["current_step"],
        current_step_index=_full_pipeline_state["current_step_index"],
        total_steps=_full_pipeline_state["total_steps"],
        step_statuses=_full_pipeline_state["step_statuses"],
        error=_full_pipeline_state["error"],
        start_time=_full_pipeline_state["start_time"],
        end_time=_full_pipeline_state["end_time"],
        duration_seconds=round(duration, 1) if duration is not None else None,
    )


def _run_step_subprocess(step: str) -> tuple[bool, str]:
    module = PIPELINE_MODULES.get(step)
    if not module:
        return False, f"Unknown step '{step}'"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    res = subprocess.run(
        [PYTHON_BIN, "-m", module],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if res.returncode != 0:
        err = (res.stderr or res.stdout or "").strip() or f"Step {step} failed"
        return False, err
    return True, (res.stdout or "").strip()


def _run_full_pipeline():
    global _full_pipeline_state
    _full_pipeline_state.update({
        "is_running": True,
        "error":      None,
        "start_time": time.time(),
        "end_time":   None,
        "current_step_index": 0,
        "step_statuses": {step: "pending" for step in PIPELINE_ORDER},
        "logs": [],
    })

    try:
        for idx, step in enumerate(PIPELINE_ORDER, 1):
            _full_pipeline_state["current_step"]       = step
            _full_pipeline_state["current_step_index"] = idx
            _full_pipeline_state["step_statuses"][step] = "running"

            ok, msg = _run_step_subprocess(step)
            _full_pipeline_state["logs"].append({"step": step, "output": msg[:2000]})

            if not ok:
                _full_pipeline_state["step_statuses"][step] = "failed"
                _full_pipeline_state["error"]      = msg
                _full_pipeline_state["is_running"] = False
                _full_pipeline_state["end_time"]   = time.time()
                return

            _full_pipeline_state["step_statuses"][step] = "completed"

        _full_pipeline_state.update({
            "current_step": None,
            "is_running":   False,
            "end_time":     time.time(),
        })
    except Exception as e:
        _full_pipeline_state.update({
            "error":      str(e),
            "is_running": False,
            "end_time":   time.time(),
        })


@app.post("/api/pipeline/run-all", tags=["Pipeline"])
def run_full_pipeline_endpoint(background_tasks: BackgroundTasks):
    if _full_pipeline_state.get("is_running"):
        return {"status": "already_running"}
    background_tasks.add_task(_run_full_pipeline)
    return {"status": "started"}


@app.post("/api/pipeline/run", tags=["Pipeline"])
def run_pipeline_step(req: RunStepRequest, background_tasks: BackgroundTasks):
    if req.step not in PIPELINE_MODULES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown step '{req.step}'. Valid: {sorted(PIPELINE_MODULES)}"
        )
    if _running_step.get(req.step):
        return {"status": "already_running", "step": req.step}
    _running_step[req.step] = True

    def _task():
        _run_step_subprocess(req.step)
        _running_step.pop(req.step, None)

    background_tasks.add_task(_task)
    return {"status": "started", "step": req.step}


@app.get("/api/pipeline/logs", tags=["Pipeline"])
def get_pipeline_logs():
    return _full_pipeline_state.get("logs", [])


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

@app.get("/api/traces", response_model=list[TraceSummary], tags=["Traces"])
def list_all_traces(limit: int = 100, offset: int = 0, error_only: bool = False):
    """Return paginated list of traces (all or error-only)."""
    if error_only:
        traces = _load_jsonl(ERROR_FILE)
    else:
        all_t    = _load_jsonl(TRACES_FILE)
        error_t  = {t["trace_id"] for t in _load_jsonl(ERROR_FILE)}
        for t in all_t:
            t.setdefault("is_error", t["trace_id"] in error_t)
            t.setdefault("flagged_by", None)
            t.setdefault("rule_tags", [])
        traces = all_t

    page = traces[offset: offset + limit]
    return [_trace_to_summary(t) for t in page]


@app.get("/api/traces/{trace_id}", response_model=TraceDetail, tags=["Traces"])
def get_trace(trace_id: str):
    """Return full detail for one trace including its facet."""
    # Check error traces first, then all traces
    error_idx = _error_trace_index()
    all_traces = {t["trace_id"]: t for t in _load_jsonl(TRACES_FILE)}

    t = error_idx.get(trace_id) or all_traces.get(trace_id)
    if not t:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")

    facet = _facet_index().get(trace_id)
    t.setdefault("is_error", trace_id in error_idx)
    t.setdefault("flagged_by", None)
    t.setdefault("rule_tags", [])

    return TraceDetail(
        trace_id=t.get("trace_id", ""),
        span_count=t.get("span_count", 0),
        total_duration_ms=t.get("total_duration_ms", 0.0),
        operations=t.get("operations", []),
        models=t.get("models", []),
        tools_called=t.get("tools_called", []),
        is_error=t.get("is_error", False),
        flagged_by=t.get("flagged_by"),
        rule_tags=t.get("rule_tags", []),
        start_time=t.get("start_time", ""),
        input_tokens=t.get("input_tokens", 0),
        output_tokens=t.get("output_tokens", 0),
        span_ids=t.get("span_ids", []),
        error_messages=t.get("error_messages", []),
        exception_types=t.get("exception_types", []),
        http_errors=t.get("http_errors", []),
        finish_reasons=t.get("finish_reasons", []),
        ae_recon_error=t.get("ae_recon_error"),
        facet=facet,
    )


@app.get("/api/error-traces", response_model=list[TraceSummary], tags=["Traces"])
def list_error_traces(limit: int = 100, offset: int = 0):
    traces = _load_jsonl(ERROR_FILE)
    return [_trace_to_summary(t) for t in traces[offset: offset + limit]]


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

@app.get("/api/clusters", response_model=list[ClusterSummary], tags=["Clusters"])
def list_clusters():
    if not CLUSTERS_FILE.exists():
        return []
    data     = _load_json(CLUSTERS_FILE)
    clusters = data.get("clusters", {})
    result   = []
    for c in sorted(clusters.values(), key=lambda x: -x["size"]):
        result.append(ClusterSummary(
            cluster_id=c["cluster_id"],
            label=c.get("label"),
            description=c.get("description"),
            is_unique_error=False,
            size=c["size"],
            dominant_issue_type=c.get("dominant_issue_type", ""),
            affected_components=c.get("affected_components", []),
            representative_trace_id=c.get("representative_trace_id", ""),
        ))
    return result


@app.get("/api/clusters/{cluster_id}", response_model=ClusterDetail, tags=["Clusters"])
def get_cluster(cluster_id: str):
    data     = _load_json(CLUSTERS_FILE)
    clusters = data.get("clusters", {})
    if cluster_id not in clusters:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found.")
    c = clusters[cluster_id]
    return ClusterDetail(
        cluster_id=c["cluster_id"],
        label=c.get("label"),
        description=c.get("description"),
        is_unique_error=False,
        size=c["size"],
        dominant_issue_type=c.get("dominant_issue_type", ""),
        affected_components=c.get("affected_components", []),
        representative_trace_id=c.get("representative_trace_id", ""),
        trace_ids=c.get("trace_ids", []),
    )


@app.get("/api/clusters/{cluster_id}/traces", response_model=list[TraceSummary], tags=["Clusters"])
def get_cluster_traces(cluster_id: str, limit: int = 50, offset: int = 0):
    data     = _load_json(CLUSTERS_FILE)
    clusters = data.get("clusters", {})
    if cluster_id not in clusters:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found.")

    trace_ids  = clusters[cluster_id].get("trace_ids", [])[offset: offset + limit]
    error_idx  = _error_trace_index()
    result     = []
    for tid in trace_ids:
        t = error_idx.get(tid)
        if t:
            result.append(_trace_to_summary(t))
    return result


# ---------------------------------------------------------------------------
# Unique errors
# ---------------------------------------------------------------------------

@app.get("/api/unique-errors", response_model=list[UniqueError], tags=["Clusters"])
def list_unique_errors(limit: int = 200, offset: int = 0):
    if not CLUSTERS_FILE.exists():
        return []
    data          = _load_json(CLUSTERS_FILE)
    unique_errors = data.get("unique_errors", [])
    page          = unique_errors[offset: offset + limit]
    return [
        UniqueError(
            trace_id=ue.get("trace_id", ""),
            issue_type=ue.get("issue_type", ""),
            summary=ue.get("summary", ""),
            component=ue.get("component", "unknown"),
            root_cause=ue.get("root_cause", ""),
        )
        for ue in page
    ]


# ---------------------------------------------------------------------------
# Triage report
# ---------------------------------------------------------------------------

@app.get("/api/triage", tags=["Triage"])
def get_triage_report():
    if not TRIAGE_REPORT.exists():
        raise HTTPException(status_code=404, detail="Triage report not found. Run triage step first.")
    return json.loads(TRIAGE_REPORT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@app.get("/api/stats", tags=["Stats"])
def get_stats():
    triage_stats: dict = {}
    if TRIAGE_REPORT.exists():
        triage_stats = json.loads(TRIAGE_REPORT.read_text(encoding="utf-8"))

    clusters_data: dict = {}
    if CLUSTERS_FILE.exists():
        clusters_data = json.loads(CLUSTERS_FILE.read_text(encoding="utf-8"))

    # Model / operation distribution across all traces
    all_traces = _load_jsonl(TRACES_FILE)
    models: dict[str, int] = {}
    ops:    dict[str, int] = {}
    for t in all_traces:
        for m in (t.get("models") or []):
            models[m] = models.get(m, 0) + 1
        for o in (t.get("operations") or []):
            ops[o] = ops.get(o, 0) + 1

    return {
        "total_traces":          triage_stats.get("total_traces", len(all_traces)),
        "error_traces":          triage_stats.get("error_traces", 0),
        "healthy_traces":        triage_stats.get("healthy_traces", 0),
        "error_rate_pct":        triage_stats.get("error_rate_pct", 0),
        "caught_by_rules":       triage_stats.get("caught_by_rules", 0),
        "caught_by_autoencoder": triage_stats.get("caught_by_autoencoder", 0),
        "error_clusters":        len(clusters_data.get("clusters", {})),
        "unique_errors":         len(clusters_data.get("unique_errors", [])),
        "rule_tag_breakdown":    triage_stats.get("rule_tag_breakdown", {}),
        "model_distribution":    dict(sorted(models.items(), key=lambda x: -x[1])),
        "operation_distribution": dict(sorted(ops.items(), key=lambda x: -x[1])),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@app.get("/api/report", tags=["Report"])
def get_report():
    if not REPORT_FILE.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet.")
    return {"report": REPORT_FILE.read_text(encoding="utf-8")}
