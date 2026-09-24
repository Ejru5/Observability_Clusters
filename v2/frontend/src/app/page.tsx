"use client";

import React, { useState, useEffect, useCallback } from "react";
import { api, Stats, ClusterSummary, UniqueError, TraceSummary, PipelineStepStatus, FullPipelineStatus } from "@/lib/api";

// ---- Icon primitives (inline SVG, no library dependency) ----
const Icon = {
  Overview: () => (
    <svg className="nav-item__icon" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="2" y="2" width="6" height="6" rx="1.5" /><rect x="10" y="2" width="6" height="6" rx="1.5" />
      <rect x="2" y="10" width="6" height="6" rx="1.5" /><rect x="10" y="10" width="6" height="6" rx="1.5" />
    </svg>
  ),
  Cluster: () => (
    <svg className="nav-item__icon" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="9" cy="9" r="2.5" /><circle cx="3" cy="4" r="1.5" /><circle cx="15" cy="4" r="1.5" />
      <circle cx="3" cy="14" r="1.5" /><circle cx="15" cy="14" r="1.5" />
      <line x1="9" y1="6.5" x2="3" y2="5.5" /><line x1="9" y1="6.5" x2="15" y2="5.5" />
      <line x1="9" y1="11.5" x2="3" y2="12.5" /><line x1="9" y1="11.5" x2="15" y2="12.5" />
    </svg>
  ),
  Unique: () => (
    <svg className="nav-item__icon" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M9 2L2 16h14L9 2z" /><line x1="9" y1="8" x2="9" y2="12" /><circle cx="9" cy="14" r="0.5" fill="currentColor" />
    </svg>
  ),
  Traces: () => (
    <svg className="nav-item__icon" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 5h14M2 9h10M2 13h12" strokeLinecap="round" />
    </svg>
  ),
  Pipeline: () => (
    <svg className="nav-item__icon" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="4" cy="9" r="2" /><circle cx="14" cy="9" r="2" />
      <circle cx="9" cy="4" r="2" /><circle cx="9" cy="14" r="2" />
      <line x1="6" y1="9" x2="12" y2="9" /><line x1="9" y1="6" x2="9" y2="12" />
    </svg>
  ),
  Play: () => (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <path d="M3 2l9 5-9 5V2z" />
    </svg>
  ),
  Close: () => (
    <svg width="12" height="12" viewBox="0 0 12 12" stroke="currentColor" strokeWidth="1.5">
      <line x1="1" y1="1" x2="11" y2="11" /><line x1="11" y1="1" x2="1" y2="11" />
    </svg>
  ),
};

type View = "overview" | "clusters" | "unique-errors" | "traces" | "pipeline";

// ---- Heatmap strip ----
function HeatmapStrip({ errorTraces }: { errorTraces: TraceSummary[] }) {
  const BUCKETS = 40;
  const counts = new Array(BUCKETS).fill(0);

  if (errorTraces.length > 0) {
    const times = errorTraces
      .map((t) => new Date(t.start_time).getTime())
      .filter((t) => !isNaN(t));
    if (times.length > 0) {
      const min = Math.min(...times);
      const max = Math.max(...times) || min + 1;
      times.forEach((t) => {
        const idx = Math.min(
          BUCKETS - 1,
          Math.floor(((t - min) / (max - min)) * BUCKETS)
        );
        counts[idx]++;
      });
    }
  }

  const maxCount = Math.max(...counts, 1);

  return (
    <div className="heatmap-strip">
      <div className="heatmap-strip__title">
        <span>Error Trace Density</span>
        <span className="text-muted text-sm">{errorTraces.length} error traces over time window</span>
      </div>
      <div className="heatmap-bar-container">
        {counts.map((c, i) => (
          <div
            key={i}
            className="heatmap-bar"
            style={{ height: `${Math.max(4, (c / maxCount) * 100)}%` }}
            title={`${c} errors`}
          />
        ))}
      </div>
    </div>
  );
}

// ---- Stat card ----
function StatCard({
  label,
  value,
  sub,
  variant,
}: {
  label: string;
  value: string | number;
  sub?: string;
  variant?: "error" | "healthy" | "warning";
}) {
  return (
    <div className="card">
      <div className="card__label">{label}</div>
      <div className={`card__value${variant ? ` card__value--${variant}` : ""}`}>
        {value}
      </div>
      {sub && <div className="card__sub">{sub}</div>}
    </div>
  );
}

// ---- Skeleton loader ----
function SkeletonCard() {
  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div className="skeleton" style={{ height: 12, width: "60%" }} />
      <div className="skeleton" style={{ height: 28, width: "40%" }} />
      <div className="skeleton" style={{ height: 10, width: "80%" }} />
    </div>
  );
}

// ---- Overview ----
function OverviewView({ stats, errorTraces, clusters }: {
  stats: Stats | null;
  errorTraces: TraceSummary[];
  clusters: ClusterSummary[];
}) {
  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Overview</h1>
        <p className="page-subtitle">Trace-level error analysis across your production AI pipeline</p>
      </div>

      <HeatmapStrip errorTraces={errorTraces} />

      <div className="stats-row">
        {stats ? (
          <>
            <StatCard label="Total Traces" value={stats.total_traces.toLocaleString()} />
            <StatCard
              label="Error Traces"
              value={stats.error_traces.toLocaleString()}
              sub={`${stats.error_rate_pct}% error rate`}
              variant="error"
            />
            <StatCard
              label="Error Clusters"
              value={stats.error_clusters}
              sub={`${stats.unique_errors} unique errors`}
              variant="warning"
            />
            <StatCard
              label="Healthy Traces"
              value={stats.healthy_traces.toLocaleString()}
              variant="healthy"
            />
          </>
        ) : (
          [1, 2, 3, 4].map((i) => <SkeletonCard key={i} />)
        )}
      </div>

      {/* Top clusters */}
      {clusters.length > 0 && (
        <>
          <div className="flex items-center justify-between mb-4">
            <h2 style={{ fontSize: 16, fontWeight: 600 }}>Top Error Clusters</h2>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {clusters.slice(0, 5).map((c) => (
              <div key={c.cluster_id} className="card" style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <span className="badge badge--error" style={{ fontVariantNumeric: "tabular-nums", minWidth: 48, justifyContent: "center" }}>
                  {c.size}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 500, fontSize: 14 }} className="truncate">{c.label || "(unlabelled)"}</div>
                  <div className="text-muted text-sm truncate">{c.description}</div>
                </div>
                <div className="flex gap-2">
                  {c.affected_components.slice(0, 2).map((comp) => (
                    <span key={comp} className="badge badge--neutral">{comp}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {stats && (
        <>
          <div className="divider" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <div className="card">
              <div className="card__label" style={{ marginBottom: 12 }}>Triage Breakdown</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div className="detail-row">
                  <span className="detail-row__key">Hard rules</span>
                  <span className="badge badge--error">{stats.caught_by_rules}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-row__key">Autoencoder</span>
                  <span className="badge badge--warning">{stats.caught_by_autoencoder}</span>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="card__label" style={{ marginBottom: 12 }}>Rule Tag Breakdown</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {Object.entries(stats.rule_tag_breakdown)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 4)
                  .map(([tag, count]) => (
                    <div key={tag} className="detail-row">
                      <span className="detail-row__key text-mono text-sm">{tag}</span>
                      <span className="badge badge--error">{count}</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ---- Clusters view ----
function ClustersView({ clusters }: { clusters: ClusterSummary[] }) {
  const [selected, setSelected] = useState<ClusterSummary | null>(null);
  const [clusterTraces, setClusterTraces] = useState<TraceSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const openCluster = async (c: ClusterSummary) => {
    setSelected(c);
    setLoading(true);
    try {
      const traces = await api.clusterTraces(c.cluster_id, 20);
      setClusterTraces(traces);
    } catch {
      setClusterTraces([]);
    }
    setLoading(false);
  };

  return (
    <div className="page" style={{ paddingRight: selected ? 500 : undefined }}>
      <div className="page-header">
        <h1 className="page-title">Error Clusters</h1>
        <p className="page-subtitle">{clusters.length} clusters — grouped by LLM facet similarity</p>
      </div>

      {clusters.length === 0 ? (
        <div className="card" style={{ textAlign: "center", padding: 40, color: "var(--content-secondary)" }}>
          No clusters yet. Run the full pipeline first.
        </div>
      ) : (
        <div className="cluster-grid">
          {clusters.map((c) => (
            <div key={c.cluster_id} className="cluster-card" onClick={() => openCluster(c)}>
              <div className="cluster-card__header">
                <span className="cluster-card__id">Cluster #{c.cluster_id}</span>
                <span className="cluster-card__size">{c.size} traces</span>
              </div>
              <div className="cluster-card__label">{c.label || "(unlabelled)"}</div>
              {c.description && (
                <div className="cluster-card__desc">{c.description}</div>
              )}
              <div className="cluster-card__footer">
                {c.affected_components.slice(0, 3).map((comp) => (
                  <span key={comp} className="badge badge--neutral">{comp}</span>
                ))}
                {c.dominant_issue_type && (
                  <span className="badge badge--error">{c.dominant_issue_type}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Detail panel */}
      {selected && (
        <div className="detail-panel">
          <button className="detail-panel__close" onClick={() => setSelected(null)} aria-label="Close">
            <Icon.Close />
          </button>
          <div style={{ marginTop: 8 }}>
            <div className="text-muted text-sm">Cluster #{selected.cluster_id}</div>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginTop: 4, letterSpacing: "-0.3px" }}>
              {selected.label}
            </h2>
            {selected.description && (
              <p className="text-muted" style={{ marginTop: 8, lineHeight: 1.6, fontSize: 13 }}>
                {selected.description}
              </p>
            )}
            <div className="flex gap-2 mt-2" style={{ flexWrap: "wrap" }}>
              {selected.affected_components.map((c) => (
                <span key={c} className="badge badge--neutral">{c}</span>
              ))}
              <span className="badge badge--error">{selected.size} traces</span>
            </div>

            <div className="divider" />

            <div className="detail-section">
              <div className="detail-section__title">Member Traces</div>
              {loading ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="skeleton" style={{ height: 56, borderRadius: 8 }} />
                  ))}
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {clusterTraces.map((t) => (
                    <div key={t.trace_id} className="card" style={{ padding: 12 }}>
                      <div className="text-mono text-sm truncate" style={{ marginBottom: 4 }}>
                        {t.trace_id}
                      </div>
                      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                        <span className="badge badge--neutral">{t.total_duration_ms.toFixed(0)}ms</span>
                        <span className="badge badge--neutral">{t.span_count} spans</span>
                        <span className={`badge badge--${t.flagged_by === "autoencoder" ? "warning" : "error"}`}>
                          {t.flagged_by || "rules"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---- Unique Errors view ----
function UniqueErrorsView({ uniqueErrors }: { uniqueErrors: UniqueError[] }) {
  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Unique Errors</h1>
        <p className="page-subtitle">
          {uniqueErrors.length} singleton traces — each represents a distinct one-off failure with no similar peers
        </p>
      </div>

      {uniqueErrors.length === 0 ? (
        <div className="card" style={{ textAlign: "center", padding: 40, color: "var(--content-secondary)" }}>
          No unique errors found.
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Trace ID</th>
                <th>Issue Type</th>
                <th>Component</th>
                <th>Summary</th>
              </tr>
            </thead>
            <tbody>
              {uniqueErrors.map((ue) => (
                <tr key={ue.trace_id}>
                  <td>
                    <span className="data-table__mono">{ue.trace_id.slice(0, 20)}...</span>
                  </td>
                  <td>
                    <span className="badge badge--error">{ue.issue_type || "—"}</span>
                  </td>
                  <td>
                    <span className="badge badge--neutral">{ue.component}</span>
                  </td>
                  <td style={{ color: "var(--content-secondary)", fontSize: 12, maxWidth: 300 }}>
                    {ue.summary}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---- Traces view ----
function TracesView() {
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorOnly, setErrorOnly] = useState(false);

  useEffect(() => {
    setLoading(true);
    (errorOnly ? api.errorTraces(100) : api.allTraces(100))
      .then(setTraces)
      .catch(() => setTraces([]))
      .finally(() => setLoading(false));
  }, [errorOnly]);

  return (
    <div className="page">
      <div className="page-header">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="page-title">All Traces</h1>
            <p className="page-subtitle">{traces.length} traces loaded</p>
          </div>
          <div className="flex gap-2">
            <button
              className={`btn btn--sm ${!errorOnly ? "btn--primary" : "btn--ghost"}`}
              onClick={() => setErrorOnly(false)}
            >All</button>
            <button
              className={`btn btn--sm ${errorOnly ? "btn--danger" : "btn--ghost"}`}
              onClick={() => setErrorOnly(true)}
            >Error only</button>
          </div>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        {loading ? (
          <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 8 }}>
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="skeleton" style={{ height: 40, borderRadius: 6 }} />
            ))}
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Trace ID</th>
                <th>Duration</th>
                <th>Spans</th>
                <th>Operations</th>
                <th>Flagged By</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {traces.map((t) => (
                <tr key={t.trace_id}>
                  <td><span className="data-table__mono">{t.trace_id.slice(0, 16)}...</span></td>
                  <td><span className="data-table__mono">{t.total_duration_ms.toFixed(0)}ms</span></td>
                  <td><span className="data-table__mono">{t.span_count}</span></td>
                  <td>
                    <span className="text-muted text-sm truncate" style={{ maxWidth: 160, display: "inline-block" }}>
                      {t.operations.join(" → ") || "—"}
                    </span>
                  </td>
                  <td>
                    {t.flagged_by ? (
                      <span className={`badge badge--${t.flagged_by === "autoencoder" ? "warning" : "error"}`}>
                        {t.flagged_by}
                      </span>
                    ) : (
                      <span className="badge badge--healthy">healthy</span>
                    )}
                  </td>
                  <td>
                    <span className={`badge badge--${t.is_error ? "error" : "healthy"}`}>
                      {t.is_error ? "error" : "ok"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ---- Pipeline view ----
function PipelineView() {
  const [steps, setSteps] = useState<PipelineStepStatus[]>([]);
  const [fullStatus, setFullStatus] = useState<FullPipelineStatus | null>(null);
  const [running, setRunning] = useState(false);

  const STEP_LABELS: Record<string, string> = {
    fetch:   "Fetch Spans",
    embed:   "Embed Traces",
    triage:  "Triage (Rules + AE)",
    facet:   "Facet Extraction (LLM)",
    cluster: "Cluster Traces",
    report:  "Generate Report",
  };

  const STEP_ICONS: Record<string, string> = {
    completed: "✓",
    running:   "◌",
    failed:    "✕",
    pending:   "○",
  };

  const refresh = useCallback(async () => {
    const [s, fs] = await Promise.all([
      api.status().catch(() => [] as PipelineStepStatus[]),
      api.fullStatus().catch(() => null),
    ]);
    setSteps(s);
    setFullStatus(fs);
    if (fs?.is_running) setRunning(true);
    else setRunning(false);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [refresh]);

  const handleRunAll = async () => {
    setRunning(true);
    await api.runAll();
    setTimeout(refresh, 1000);
  };

  const statusMap = fullStatus?.step_statuses || {};

  return (
    <div className="page">
      <div className="page-header">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="page-title">Pipeline</h1>
            <p className="page-subtitle">6-step trace-level, LLM-first pipeline</p>
          </div>
          <button
            className="btn btn--primary"
            onClick={handleRunAll}
            disabled={running}
          >
            <Icon.Play />
            {running ? "Running…" : "Run All Steps"}
          </button>
        </div>
      </div>

      {fullStatus?.error && (
        <div className="card" style={{ borderColor: "rgba(255,69,58,0.3)", background: "rgba(255,69,58,0.08)", marginBottom: 16 }}>
          <div className="badge badge--error" style={{ marginBottom: 8 }}>Pipeline Error</div>
          <div className="text-mono text-sm" style={{ color: "var(--accent-error)" }}>{fullStatus.error}</div>
        </div>
      )}

      {fullStatus?.duration_seconds !== null && fullStatus?.duration_seconds !== undefined && (
        <div className="card" style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 12 }}>
          <span className="badge badge--healthy">Completed</span>
          <span className="text-muted text-sm">Last run: {fullStatus.duration_seconds}s</span>
        </div>
      )}

      <div className="pipeline-steps">
        {["fetch", "embed", "triage", "facet", "cluster", "report"].map((step, idx) => {
          const done    = steps.find((s) => s.step === step)?.done;
          const status  = statusMap[step] || (done ? "completed" : "pending");
          const icon    = STEP_ICONS[status] || "○";
          const file    = steps.find((s) => s.step === step)?.file || "";

          return (
            <div key={step} className="pipeline-step">
              <div className={`pipeline-step__icon pipeline-step__icon--${status === "completed" || (done && status === "pending") ? "done" : status}`}>
                {status === "running" ? (
                  <span style={{ animation: "shimmer 1s infinite" }}>◌</span>
                ) : (
                  done ? "✓" : icon
                )}
              </div>
              <div className="pipeline-step__body">
                <div className="pipeline-step__name">
                  {idx + 1}. {STEP_LABELS[step] || step}
                </div>
                {file && <div className="pipeline-step__file">{file.split("\\").pop()}</div>}
              </div>
              <div className="flex gap-2 items-center">
                {done && <span className="badge badge--healthy">Done</span>}
                {status === "running" && <span className="badge badge--blue">Running</span>}
                {status === "failed"  && <span className="badge badge--error">Failed</span>}
                <button
                  className="btn btn--ghost btn--sm"
                  onClick={() => api.runStep(step)}
                  disabled={running}
                >
                  Run
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- Main app ----
export default function DashboardPage() {
  const [view, setView] = useState<View>("overview");
  const [stats, setStats]           = useState<Stats | null>(null);
  const [clusters, setClusters]     = useState<ClusterSummary[]>([]);
  const [uniqueErrors, setUniqueErrors] = useState<UniqueError[]>([]);
  const [errorTraces, setErrorTraces]   = useState<TraceSummary[]>([]);
  const [pipelineOk, setPipelineOk]     = useState(true);

  useEffect(() => {
    Promise.allSettled([
      api.stats().then(setStats),
      api.clusters().then(setClusters),
      api.uniqueErrors().then(setUniqueErrors),
      api.errorTraces(200).then(setErrorTraces),
    ]).then((results) => {
      const allFailed = results.every((r) => r.status === "rejected");
      if (allFailed) setPipelineOk(false);
    });
  }, []);

  const navItems: { id: View; label: string; icon: () => React.ReactNode; badge?: number }[] = [
    { id: "overview",      label: "Overview",       icon: Icon.Overview },
    { id: "clusters",      label: "Error Clusters", icon: Icon.Cluster,  badge: clusters.length },
    { id: "unique-errors", label: "Unique Errors",  icon: Icon.Unique,   badge: uniqueErrors.length },
    { id: "traces",        label: "All Traces",     icon: Icon.Traces },
    { id: "pipeline",      label: "Pipeline",       icon: Icon.Pipeline },
  ];

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="app-header">
        <div className="app-header__title">
          <div className={`app-header__dot${pipelineOk && stats && stats.error_rate_pct < 5 ? " app-header__dot--healthy" : ""}`} />
          Observability Clusters v2
        </div>
        <div className="app-header__actions">
          {stats && (
            <span className="text-muted text-sm">
              {stats.total_traces.toLocaleString()} traces · {stats.error_rate_pct}% error rate
            </span>
          )}
        </div>
      </header>

      {/* Sidebar */}
      <nav className="app-sidebar">
        <div className="app-sidebar__section">Navigation</div>
        {navItems.map((item) => (
          <button
            key={item.id}
            className={`nav-item${view === item.id ? " active" : ""}`}
            onClick={() => setView(item.id)}
          >
            <item.icon />
            {item.label}
            {item.badge !== undefined && item.badge > 0 && (
              <span className="nav-item__badge">{item.badge}</span>
            )}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="app-content">
        {view === "overview"      && <OverviewView stats={stats} errorTraces={errorTraces} clusters={clusters} />}
        {view === "clusters"      && <ClustersView clusters={clusters} />}
        {view === "unique-errors" && <UniqueErrorsView uniqueErrors={uniqueErrors} />}
        {view === "traces"        && <TracesView />}
        {view === "pipeline"      && <PipelineView />}
      </main>
    </div>
  );
}
