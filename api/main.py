"""
FastAPI backend for the Observability Clusters dashboard.

Reads pre-computed cluster JSON + span data and exposes REST endpoints.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
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
DATA_DIR        = Path("data")
MACRO_FILE      = DATA_DIR / "macro_clusters.json"
MICRO_FILE      = DATA_DIR / "micro_clusters.json"
SPAN_FILE       = DATA_DIR / "spans.jsonl"
REPORT_FILE     = DATA_DIR / "report.md"
TRIAGE_SCORES   = DATA_DIR / "triage_scores.json"
TRIAGE_REPORT   = DATA_DIR / "triage_report.json"
PROB_SPAN_FILE  = DATA_DIR / "problematic_spans.jsonl"
PROB_EMB_FILE   = DATA_DIR / "problematic_embeddings.npy"

ROOT_DIR        = Path(__file__).parent.parent
VENV_PYTHON     = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
PYTHON_BIN      = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

app = FastAPI(
    title="Observability Clusters API",
    description="REST API for two-level embedding-based span clustering on Mistral Observability data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{path.name} not found. Run the pipeline steps first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_spans() -> dict[str, dict]:
    spans: dict[str, dict] = {}
    if SPAN_FILE.exists():
        for line in SPAN_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                s = json.loads(line)
                spans[s["span_id"]] = s
    if PROB_SPAN_FILE.exists():
        for line in PROB_SPAN_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                s = json.loads(line)
                spans[s["span_id"]] = s
    return spans


def _to_span_detail(s: dict) -> SpanDetail:
    return SpanDetail(
        span_id=s.get("span_id", ""),
        trace_id=s.get("trace_id", ""),
        operation_name=s.get("operation_name", ""),
        model=s.get("model", ""),
        tool_called=s.get("tool_called", ""),
        duration_ms=s.get("duration_ms"),
        status=s.get("status", ""),
        error_message=s.get("error_message", ""),
        exception_type=s.get("exception_type", ""),
        input_summary=s.get("input_summary", ""),
        output_summary=s.get("output_summary", ""),
        input_tokens=s.get("input_tokens"),
        output_tokens=s.get("output_tokens"),
        timestamp=str(s.get("timestamp", "")),
        is_problematic=bool(s.get("is_problematic", False)),
        problem_types=list(s.get("problem_types", [])),
        lof_score=s.get("lof_score"),
        ae_recon_error=s.get("ae_recon_error"),
        triage_algorithm=s.get("triage_algorithm"),
    )


# ---------------------------------------------------------------------------
# Pipeline status helper
# ---------------------------------------------------------------------------

PIPELINE_FILES = {
    "fetch":         DATA_DIR / "raw_spans.jsonl",
    "embed":         DATA_DIR / "embeddings.npy",
    "triage":        TRIAGE_SCORES,
    "apply_triage":  PROB_SPAN_FILE,
    "macro":         MACRO_FILE,
    "micro":         MICRO_FILE,
    "label":         MACRO_FILE,   # label updates these in-place
    "report":        REPORT_FILE,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PipelineStatus(BaseModel):
    step: str
    done: bool
    file: str


class FullPipelineStatusResponse(BaseModel):
    is_running: bool
    current_step: str | None
    current_step_index: int
    total_steps: int
    step_statuses: dict[str, str]
    error: str | None
    start_time: float | None
    end_time: float | None
    duration_seconds: float | None


class ClusterSummary(BaseModel):
    cluster_id: str
    label: str | None
    description: str | None
    size: int
    is_noise: bool
    micro_cluster_count: int


class SpanDetail(BaseModel):
    span_id: str
    trace_id: str
    operation_name: str
    model: str
    tool_called: str
    duration_ms: float | None
    status: str
    error_message: str
    exception_type: str
    input_summary: str
    output_summary: str
    input_tokens: int | None
    output_tokens: int | None
    timestamp: str
    # Triage fields (present only after triage step has run)
    is_problematic: bool = False
    problem_types: list[str] = []
    lof_score: float | None = None
    ae_recon_error: float | None = None
    triage_algorithm: str | None = None


class TriageScore(BaseModel):
    span_id: str
    rule_fired: bool
    rule_types: list[str]
    lof_score: float
    is_anomaly_lof: bool
    is_problematic_lof: bool
    problem_types_lof: list[str]
    ae_reconstruction_error: float
    is_anomaly_ae: bool
    is_problematic_ae: bool
    problem_types_ae: list[str]


class SelectAlgoRequest(BaseModel):
    algorithm: str   # "lof" | "ae"


class MacroClusterDetail(BaseModel):
    cluster_id: str
    label: str | None
    description: str | None
    size: int
    is_noise: bool
    span_ids: list[str]
    micro_clusters: dict[str, Any]


class ClusterPoint(BaseModel):
    span_id: str
    cluster_id: str
    cluster_label: str
    x: float
    y: float
    operation_name: str
    model: str
    status: str
    duration_ms: float | None
    input_summary: str
    is_noise: bool


class RunStepRequest(BaseModel):
    step: str   # "fetch"|"embed"|"triage"|"apply_triage"|"macro"|"micro"|"label"|"report"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Observability Clusters API"}


@app.get("/api/status", response_model=list[PipelineStatus], tags=["Pipeline"])
def get_pipeline_status():
    """Return whether each pipeline step has been completed."""
    steps = [
        ("fetch",        DATA_DIR / "raw_spans.jsonl"),
        ("embed",        DATA_DIR / "embeddings.npy"),
        ("triage",       TRIAGE_SCORES),
        ("apply_triage", PROB_SPAN_FILE),
        ("macro",        MACRO_FILE),
        ("micro",        MICRO_FILE),
        ("report",       REPORT_FILE),
    ]
    result = []
    for step, path in steps:
        done = path.exists() and path.stat().st_size > 0
        result.append(PipelineStatus(step=step, done=done, file=str(path)))

    # "label" is done if macro_clusters.json has non-null labels
    try:
        macro = json.loads(MACRO_FILE.read_text(encoding="utf-8")) if MACRO_FILE.exists() else {}
        labelled = any(c.get("label") for c in macro.values() if not c.get("is_noise"))
    except Exception:
        labelled = False
    result.insert(6, PipelineStatus(step="label", done=labelled, file=str(MACRO_FILE)))

    return result


@app.get("/api/clusters", response_model=list[ClusterSummary], tags=["Clusters"])
def list_clusters():
    """Return all macro clusters with summary info."""
    if not MACRO_FILE.exists():
        return []
    macro = _load_json(MACRO_FILE)
    micro = _load_json(MICRO_FILE) if MICRO_FILE.exists() else {}

    result = []
    for cid, cluster in macro.items():
        sub = micro.get(cid, {})
        micro_count = sum(1 for c in sub.values() if not c.get("is_noise"))
        result.append(ClusterSummary(
            cluster_id=cid,
            label=cluster.get("label"),
            description=cluster.get("description"),
            size=cluster["size"],
            is_noise=cluster["is_noise"],
            micro_cluster_count=micro_count,
        ))

    # Sort: real clusters by size desc, noise last
    result.sort(key=lambda x: (x.is_noise, -x.size))
    return result


POINTS_FILE = DATA_DIR / "points_2d.json"

def _get_or_compute_points() -> list[dict[str, Any]]:
    if POINTS_FILE.exists():
        try:
            return json.loads(POINTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Use problematic spans + their embeddings if triage has run, else fall back to all spans
    if PROB_SPAN_FILE.exists() and PROB_EMB_FILE.exists() and PROB_SPAN_FILE.stat().st_size > 0:
        active_span_file = PROB_SPAN_FILE
        active_emb_file  = PROB_EMB_FILE
    else:
        active_span_file = SPAN_FILE
        active_emb_file  = DATA_DIR / "embeddings.npy"

    if not (active_emb_file.exists() and active_span_file.exists() and MACRO_FILE.exists()):
        return []

    import numpy as np

    spans = [json.loads(l) for l in active_span_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    macro = json.loads(MACRO_FILE.read_text(encoding="utf-8"))
    emb   = np.load(active_emb_file)

    if len(spans) == 0 or len(emb) == 0:
        return []

    # Build triage score index for enriching points
    triage_by_id: dict[str, dict] = {}
    if TRIAGE_SCORES.exists():
        for rec in json.loads(TRIAGE_SCORES.read_text(encoding="utf-8")):
            triage_by_id[rec["span_id"]] = rec

    # UMAP 2D projection
    try:
        import umap
        n_neighbors = min(15, max(2, len(emb) - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords_2d = reducer.fit_transform(emb)
    except Exception:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(emb)

    min_val, max_val = coords_2d.min(axis=0), coords_2d.max(axis=0)
    range_val = np.where(max_val - min_val == 0, 1.0, max_val - min_val)
    norm_coords = ((coords_2d - min_val) / range_val) * 200 - 100

    span_to_cluster: dict[str, str] = {}
    cluster_labels: dict[str, str]  = {}
    for cid, c in macro.items():
        label = c.get("label") or ("Noise Spans" if c.get("is_noise") else f"Cluster {cid}")
        cluster_labels[cid] = label
        for sid in c.get("span_ids", []):
            span_to_cluster[sid] = cid

    points = []
    for i, s in enumerate(spans):
        sid     = s.get("span_id", f"span_{i}")
        cid     = span_to_cluster.get(sid, "-1")
        is_noise = (cid == "-1") or macro.get(cid, {}).get("is_noise", False)
        tr      = triage_by_id.get(sid, {})

        # Determine active algo from span metadata
        algo = s.get("triage_algorithm", "")
        prob_types_key = f"problem_types_{algo}" if algo else "problem_types_lof"
        types = s.get("problem_types") or tr.get(prob_types_key, [])

        points.append({
            "span_id":        sid,
            "cluster_id":     cid,
            "cluster_label":  cluster_labels.get(cid, "Noise Spans"),
            "x":              round(float(norm_coords[i, 0]), 2),
            "y":              round(float(norm_coords[i, 1]), 2),
            "operation_name": s.get("operation_name", ""),
            "model":          s.get("model", ""),
            "status":         s.get("status", "OK"),
            "duration_ms":    s.get("duration_ms"),
            "input_summary":  (s.get("input_summary") or "")[:100],
            "is_noise":       is_noise,
            # Triage enrichment
            "is_problematic": s.get("is_problematic", bool(tr.get("is_problematic_lof"))),
            "problem_types":  list(types),
            "lof_score":      tr.get("lof_score"),
            "ae_recon_error": tr.get("ae_reconstruction_error"),
        })

    POINTS_FILE.write_text(json.dumps(points, indent=2), encoding="utf-8")
    return points


@app.get("/api/cluster-points", tags=["Clusters"])
def get_cluster_points():
    """Return 2D projected scatter points for problematic span embeddings across macro clusters."""
    return _get_or_compute_points()


# ---------------------------------------------------------------------------
# Triage endpoints
# ---------------------------------------------------------------------------

@app.get("/api/triage", tags=["Triage"])
def get_triage_report():
    """Return the triage report with problem counts and type breakdown."""
    if not TRIAGE_REPORT.exists():
        raise HTTPException(
            status_code=404,
            detail="Triage report not found. Run the triage pipeline step first."
        )
    return json.loads(TRIAGE_REPORT.read_text(encoding="utf-8"))


@app.get("/api/triage/scores", tags=["Triage"])
def get_triage_scores(limit: int = 200, offset: int = 0):
    """Return per-span triage scores (LOF + AE) with pagination."""
    if not TRIAGE_SCORES.exists():
        raise HTTPException(
            status_code=404,
            detail="Triage scores not found. Run the triage pipeline step first."
        )
    all_scores = json.loads(TRIAGE_SCORES.read_text(encoding="utf-8"))
    return all_scores[offset: offset + limit]


@app.post("/api/triage/select", tags=["Triage"])
def select_triage_algorithm(req: SelectAlgoRequest, background_tasks: BackgroundTasks):
    """
    Switch the active triage algorithm (lof | ae).
    Runs apply_triage in the background which filters spans and invalidates
    downstream caches so the next /api/cluster-points call re-computes UMAP.
    """
    valid = {"lof", "ae"}
    algo  = req.algorithm.lower().strip()
    if algo not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm '{req.algorithm}'. Valid: {sorted(valid)}"
        )
    if not TRIAGE_SCORES.exists():
        raise HTTPException(
            status_code=400,
            detail="Triage scores not found. Run the triage step first."
        )

    def _apply_and_cluster():
        subprocess.run([PYTHON_BIN, "-m", "src.apply_triage", "--algo", algo], cwd=ROOT_DIR)
        subprocess.run([PYTHON_BIN, "-m", "src.cluster_macro"], cwd=ROOT_DIR)
        subprocess.run([PYTHON_BIN, "-m", "src.cluster_micro"], cwd=ROOT_DIR)
        subprocess.run([PYTHON_BIN, "-m", "src.label"], cwd=ROOT_DIR)
        subprocess.run([PYTHON_BIN, "-m", "src.report"], cwd=ROOT_DIR)

    background_tasks.add_task(_apply_and_cluster)
    return {"status": "started", "algorithm": algo}



@app.get("/api/clusters/{cluster_id}", response_model=MacroClusterDetail, tags=["Clusters"])
def get_cluster(cluster_id: str):
    """Return full detail for one macro cluster including micro-clusters."""
    if not MACRO_FILE.exists():
        raise HTTPException(status_code=404, detail="Clusters not found. Please run the clustering pipeline.")
    macro = _load_json(MACRO_FILE)
    micro = _load_json(MICRO_FILE) if MICRO_FILE.exists() else {}

    if cluster_id not in macro:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found.")

    cluster = macro[cluster_id]
    return MacroClusterDetail(
        cluster_id=cluster_id,
        label=cluster.get("label"),
        description=cluster.get("description"),
        size=cluster["size"],
        is_noise=cluster["is_noise"],
        span_ids=cluster["span_ids"],
        micro_clusters=micro.get(cluster_id, {}),
    )


@app.get("/api/clusters/{cluster_id}/spans", response_model=list[SpanDetail], tags=["Spans"])
def get_cluster_spans(cluster_id: str, limit: int = 50):
    """Return up to `limit` spans for a macro cluster."""
    if not MACRO_FILE.exists():
        raise HTTPException(status_code=404, detail="Clusters not found. Please run the clustering pipeline.")
    macro = _load_json(MACRO_FILE)
    if cluster_id not in macro:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found.")

    all_spans = _load_spans()
    span_ids = macro[cluster_id]["span_ids"][:limit]

    spans = []
    for sid in span_ids:
        s = all_spans.get(sid)
        if s:
            spans.append(_to_span_detail(s))
    return spans


@app.get("/api/spans", response_model=list[SpanDetail], tags=["Spans"])
def list_spans(limit: int = 100, offset: int = 0):
    """Return a paginated list of all normalised spans."""
    all_spans = _load_spans()
    items = list(all_spans.values())[offset: offset + limit]
    return [_to_span_detail(s) for s in items]


@app.get("/api/spans/{span_id}", response_model=SpanDetail, tags=["Spans"])
def get_span(span_id: str):
    """Return a single span by ID."""
    all_spans = _load_spans()
    s = all_spans.get(span_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Span '{span_id}' not found.")
    return _to_span_detail(s)


@app.get("/api/report", tags=["Report"])
def get_report():
    """Return the markdown report as plain text."""
    if not REPORT_FILE.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet. Run the pipeline first.")
    return {"report": REPORT_FILE.read_text(encoding="utf-8")}


@app.get("/api/stats", tags=["Stats"])
def get_stats():
    """Return high-level statistics about the current clustering run."""
    all_spans = _load_spans()
    models: dict[str, int] = {}
    ops: dict[str, int] = {}
    for s in all_spans.values():
        m = s.get("model") or "unknown"
        models[m] = models.get(m, 0) + 1
        op = s.get("operation_name") or "unknown"
        ops[op] = ops.get(op, 0) + 1

    model_dist = dict(sorted(models.items(), key=lambda x: -x[1]))
    op_dist = dict(sorted(ops.items(), key=lambda x: -x[1]))

    if not MACRO_FILE.exists():
        return {
            "total_spans": len(all_spans),
            "macro_clusters": 0,
            "micro_clusters": 0,
            "noise_spans": 0,
            "labelled": 0,
            "model_distribution": model_dist,
            "operation_distribution": op_dist,
        }

    try:
        macro = _load_json(MACRO_FILE)
    except Exception:
        macro = {}

    micro = json.loads(MICRO_FILE.read_text(encoding="utf-8")) if MICRO_FILE.exists() else {}

    total_spans   = len(all_spans)
    noise_spans   = sum(c["size"] for c in macro.values() if c.get("is_noise")) if macro else 0
    macro_count   = sum(1 for c in macro.values() if not c.get("is_noise")) if macro else 0
    labelled      = sum(1 for c in macro.values() if c.get("label") and not c.get("is_noise")) if macro else 0

    total_micro = sum(
        sum(1 for sc in sub.values() if not sc.get("is_noise"))
        for sub in micro.values()
    )

    return {
        "total_spans":    total_spans,
        "macro_clusters": macro_count,
        "micro_clusters": total_micro,
        "noise_spans":    noise_spans,
        "labelled":       labelled,
        "model_distribution": model_dist,
        "operation_distribution": op_dist,
    }


# background pipeline runner
_running_step: dict[str, bool] = {}

PIPELINE_ORDER = ["fetch", "embed", "triage", "apply_triage", "macro", "micro", "label", "report"]

_full_pipeline_state: dict[str, Any] = {
    "is_running": False,
    "current_step": None,
    "current_step_index": 0,
    "total_steps": 8,
    "step_statuses": {step: "pending" for step in PIPELINE_ORDER},
    "error": None,
    "start_time": None,
    "end_time": None,
}


def _run_pipeline_step(step: str):
    invalidate_on = {"embed", "triage", "apply_triage", "macro", "label"}
    if step in invalidate_on and POINTS_FILE.exists():
        try:
            POINTS_FILE.unlink()
        except Exception:
            pass
    module_map = {
        "fetch":        "src.fetch_spans",
        "embed":        "src.embed",
        "triage":       "src.triage",
        "apply_triage": "src.apply_triage",
        "macro":        "src.cluster_macro",
        "micro":        "src.cluster_micro",
        "label":        "src.label",
        "report":       "src.report",
    }
    module = module_map.get(step)
    if not module:
        return
    subprocess.run(
        [PYTHON_BIN, "-m", module],
        cwd=ROOT_DIR,
    )
    _running_step.pop(step, None)


def _run_full_pipeline():
    global _full_pipeline_state
    _full_pipeline_state["is_running"] = True
    _full_pipeline_state["error"] = None
    _full_pipeline_state["start_time"] = time.time()
    _full_pipeline_state["end_time"] = None
    _full_pipeline_state["current_step_index"] = 0
    _full_pipeline_state["step_statuses"] = {step: "pending" for step in PIPELINE_ORDER}

    if POINTS_FILE.exists():
        try:
            POINTS_FILE.unlink()
        except Exception:
            pass

    module_map = {
        "fetch":        "src.fetch_spans",
        "embed":        "src.embed",
        "triage":       "src.triage",
        "apply_triage": "src.apply_triage",
        "macro":        "src.cluster_macro",
        "micro":        "src.cluster_micro",
        "label":        "src.label",
        "report":       "src.report",
    }

    try:
        for idx, step in enumerate(PIPELINE_ORDER, 1):
            _full_pipeline_state["current_step"] = step
            _full_pipeline_state["current_step_index"] = idx
            _full_pipeline_state["step_statuses"][step] = "running"
            
            module = module_map[step]
            res = subprocess.run(
                [PYTHON_BIN, "-m", module],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
            )
            
            if res.returncode != 0:
                err_msg = (res.stderr or res.stdout or "").strip() or f"Step {step} exited with code {res.returncode}"
                _full_pipeline_state["step_statuses"][step] = "failed"
                _full_pipeline_state["error"] = err_msg
                _full_pipeline_state["is_running"] = False
                _full_pipeline_state["end_time"] = time.time()
                return

            _full_pipeline_state["step_statuses"][step] = "completed"

        _full_pipeline_state["current_step"] = None
        _full_pipeline_state["is_running"] = False
        _full_pipeline_state["end_time"] = time.time()

        if POINTS_FILE.exists():
            try:
                POINTS_FILE.unlink()
            except Exception:
                pass

    except Exception as e:
        _full_pipeline_state["error"] = str(e)
        _full_pipeline_state["is_running"] = False
        _full_pipeline_state["end_time"] = time.time()


@app.post("/api/pipeline/run", tags=["Pipeline"])
def run_pipeline_step(req: RunStepRequest, background_tasks: BackgroundTasks):
    """Trigger a pipeline step in the background."""
    valid = {"fetch", "embed", "triage", "apply_triage", "macro", "micro", "label", "report"}
    if req.step not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown step '{req.step}'. Valid: {sorted(valid)}")
    if _running_step.get(req.step):
        return {"status": "already_running", "step": req.step}
    _running_step[req.step] = True
    background_tasks.add_task(_run_pipeline_step, req.step)
    return {"status": "started", "step": req.step}


@app.get("/api/pipeline/full-status", response_model=FullPipelineStatusResponse, tags=["Pipeline"])
def get_full_pipeline_status():
    """Return status of the end-to-end pipeline execution."""
    start = _full_pipeline_state.get("start_time")
    end = _full_pipeline_state.get("end_time") or (time.time() if _full_pipeline_state.get("is_running") else None)
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


@app.post("/api/pipeline/run-all", tags=["Pipeline"])
def run_full_pipeline_endpoint(background_tasks: BackgroundTasks):
    """Trigger the entire pipeline end-to-end in the background."""
    if _full_pipeline_state.get("is_running"):
        return {"status": "already_running"}
    
    background_tasks.add_task(_run_full_pipeline)
    return {"status": "started"}

