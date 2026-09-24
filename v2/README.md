# v2 — New Trace-Level, LLM-First Pipeline

## Structure

```
v2/
├── backend/     Python pipeline (FastAPI + 6-step pipeline)
└── frontend/    Next.js dashboard (Apple HIG dark-mode design)
```

---

## Backend

### Setup
```bash
cd v2/backend
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

### Run pipeline
```bash
python -m src.fetch_spans      # Step 1: fetch raw spans
python -m src.embed            # Step 2: normalise → rollup traces → embed
python -m src.triage           # Step 3: hard rules + AE bifurcation
python -m src.facet_extract    # Step 4: LLM facet per error trace
python -m src.cluster_traces   # Step 5: cosine similarity clustering + LLM labels
python -m src.report           # Step 6: generate report.md
```

### Run API
```bash
uvicorn api.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## Frontend

### Setup
```bash
cd v2/frontend
npm install
npm run dev       # http://localhost:3000
```

---

## Key architectural differences from v1

| | v1 | v2 |
|---|---|---|
| Unit | Spans | **Traces** |
| Triage gate | Rules + LOF + AE | **Rules + AE (high-recall, P75)** |
| Clustering | UMAP + HDBSCAN | **LLM facet → cosine similarity** |
| Singletons | Noise bin | **Unique Errors section** |
| LOF | Yes | **Removed** |
