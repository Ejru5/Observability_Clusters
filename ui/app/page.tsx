"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  ChartBar,
  Graph,
  ArrowClockwise,
  CaretRight,
  WarningCircle,
  CheckCircle,
  Circle,
  Cpu,
  Rows,
  Lightning,
  MagnifyingGlass,
  X,
  Play,
} from "@phosphor-icons/react";
import {
  fetchClusters,
  fetchStats,
  fetchStatus,
  fetchCluster,
  fetchClusterSpans,
  fetchClusterPoints,
  fetchTriageReport,
  selectTriageAlgorithm,
  runStep,
  runFullPipeline,
  fetchFullPipelineStatus,
  type ClusterSummary,
  type Stats,
  type PipelineStatus,
  type MacroClusterDetail,
  type SpanDetail,
  type ClusterPoint,
  type TriageReport,
  type FullPipelineStatus,
} from "@/lib/api";
import { ClusterCharts } from "@/components/ClusterCharts";
import { ClusterScatterPlot } from "@/components/ClusterScatterPlot";

// ---------------------------------------------------------------------------
// Colour palette per cluster index
// ---------------------------------------------------------------------------
const CLUSTER_COLORS = [
  "#00e5a0", "#00b8ff", "#a78bfa", "#f59e0b",
  "#f87171", "#34d399", "#60a5fa", "#fb923c",
];
function clusterColor(id: string): string {
  const n = parseInt(id.replace("-", ""), 10);
  return CLUSTER_COLORS[Math.abs(n) % CLUSTER_COLORS.length];
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusDot({ done }: { done: boolean }) {
  return done ? (
    <CheckCircle size={14} weight="fill" className="status-ok" />
  ) : (
    <Circle size={14} weight="regular" style={{ color: "var(--muted)" }} />
  );
}

function PipelineProgressBanner({
  status,
  onClose,
  onRunAgain,
}: {
  status: FullPipelineStatus;
  onClose: () => void;
  onRunAgain: () => void;
}) {
  const steps = ["fetch", "embed", "triage", "apply_triage", "macro", "micro", "label", "report"];
  const isRunning = status.is_running;
  const isFailed = !!status.error;
  const isCompleted = !isRunning && !isFailed && status.current_step_index >= status.total_steps;

  const pct = status.total_steps > 0
    ? Math.round((status.current_step_index / status.total_steps) * 100)
    : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      style={{
        marginBottom: 28,
        padding: "20px 24px",
        borderRadius: 20,
        background: isFailed
          ? "linear-gradient(135deg, rgba(248,113,113,0.12) 0%, rgba(20,10,12,0.95) 100%)"
          : isCompleted
            ? "linear-gradient(135deg, rgba(0,229,160,0.12) 0%, rgba(8,20,16,0.95) 100%)"
            : "linear-gradient(135deg, rgba(0,184,255,0.12) 0%, rgba(10,16,24,0.95) 100%)",
        border: `1px solid ${isFailed
            ? "rgba(248,113,113,0.3)"
            : isCompleted
              ? "rgba(0,229,160,0.3)"
              : "rgba(0,184,255,0.3)"
          }`,
        boxShadow: "0 8px 32px rgba(0,0,0,0.35)",
      }}
    >
      {/* Top Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {isRunning && (
            <div style={{
              width: 10, height: 10, borderRadius: "50%",
              background: "#00b8ff", boxShadow: "0 0 10px #00b8ff",
              animation: "pulse 1.2s infinite ease-in-out",
            }} />
          )}
          {isCompleted && (
            <CheckCircle size={18} weight="fill" style={{ color: "var(--accent)" }} />
          )}
          {isFailed && (
            <WarningCircle size={18} weight="fill" style={{ color: "#f87171" }} />
          )}
          <span style={{ fontSize: 14, fontWeight: 700, letterSpacing: "-0.01em" }}>
            {isRunning
              ? `Running End-to-End Pipeline — Step ${status.current_step_index} of ${status.total_steps}: ${status.current_step}`
              : isCompleted
                ? `End-to-End Pipeline Completed Successfully!`
                : `Pipeline Execution Failed`}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {status.duration_seconds != null && (
            <span style={{ fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: "var(--muted)" }}>
              {status.duration_seconds}s
            </span>
          )}
          {!isRunning && (
            <button
              onClick={onClose}
              style={{
                background: "transparent", border: "none", color: "var(--muted)",
                cursor: "pointer", display: "flex", alignItems: "center",
              }}
            >
              <X size={16} />
            </button>
          )}
        </div>
      </div>

      {/* Progress Track */}
      <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 999, overflow: "hidden", marginBottom: 16 }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.3 }}
          style={{
            height: "100%",
            background: isFailed
              ? "#f87171"
              : isCompleted
                ? "linear-gradient(90deg, #00e5a0, #34d399)"
                : "linear-gradient(90deg, #00b8ff, #00e5a0)",
            borderRadius: 999,
          }}
        />
      </div>

      {/* Steps Visual Tracker */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))", gap: 8 }}>
        {steps.map((step) => {
          const st = status.step_statuses[step] ?? "pending";
          const isCurrent = status.current_step === step;
          return (
            <div
              key={step}
              style={{
                padding: "8px 10px",
                borderRadius: 10,
                background: isCurrent
                  ? "rgba(0,184,255,0.15)"
                  : st === "completed"
                    ? "rgba(0,229,160,0.08)"
                    : st === "failed"
                      ? "rgba(248,113,113,0.12)"
                      : "rgba(255,255,255,0.02)",
                border: `1px solid ${isCurrent
                    ? "#00b8ff"
                    : st === "completed"
                      ? "rgba(0,229,160,0.25)"
                      : st === "failed"
                        ? "rgba(248,113,113,0.3)"
                        : "var(--border)"
                  }`,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              {st === "completed" ? (
                <CheckCircle size={13} weight="fill" style={{ color: "var(--accent)", flexShrink: 0 }} />
              ) : st === "running" ? (
                <ArrowClockwise size={13} style={{ color: "#00b8ff", animation: "spin 1s linear infinite", flexShrink: 0 }} />
              ) : st === "failed" ? (
                <WarningCircle size={13} weight="fill" style={{ color: "#f87171", flexShrink: 0 }} />
              ) : (
                <Circle size={13} style={{ color: "var(--muted)", flexShrink: 0 }} />
              )}
              <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", fontWeight: isCurrent ? 700 : 500 }}>
                {step}
              </span>
            </div>
          );
        })}
      </div>

      {/* Error Message */}
      {status.error && (
        <div style={{ marginTop: 14, fontSize: 12, color: "#f87171", fontFamily: "'JetBrains Mono', monospace", background: "rgba(248,113,113,0.08)", padding: "10px 14px", borderRadius: 8 }}>
          {status.error}
        </div>
      )}

      {/* Completion Notification */}
      {isCompleted && (
        <div style={{ marginTop: 14, fontSize: 12, color: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span>✨ All pipeline steps finished! Charts and cluster metrics updated.</span>
          <button
            onClick={onRunAgain}
            style={{
              padding: "4px 12px", borderRadius: 8, background: "rgba(0,229,160,0.15)",
              border: "1px solid rgba(0,229,160,0.3)", color: "var(--accent)", fontSize: 11, fontWeight: 700,
              cursor: "pointer",
            }}
          >
            Run Again
          </button>
        </div>
      )}
    </motion.div>
  );
}

function PipelinePanel({
  statuses,
  onRefresh,
  onRunFullPipeline,
  isFullRunning,
  fullStatus,
}: {
  statuses: PipelineStatus[];
  onRefresh?: () => void;
  onRunFullPipeline: () => void;
  isFullRunning: boolean;
  fullStatus: FullPipelineStatus | null;
}) {
  const [running, setRunning] = useState<string | null>(null);
  const steps = ["fetch", "embed", "triage", "apply_triage", "macro", "micro", "label", "report"];

  async function trigger(step: string) {
    setRunning(step);
    try {
      await runStep(step);
    } catch (err) {
      console.error(`Error triggering step ${step}:`, err);
    }

    let count = 0;
    const interval = setInterval(async () => {
      count++;
      if (onRefresh) onRefresh();
      if (count > 20) {
        clearInterval(interval);
        setRunning(null);
      }
    }, 2500);

    setTimeout(() => {
      clearInterval(interval);
      setRunning(null);
      if (onRefresh) onRefresh();
    }, 15000);
  }

  return (
    <div className="bezel" style={{ height: "100%" }}>
      <div className="bezel-inner" style={{ padding: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
          <Lightning size={16} weight="fill" style={{ color: "var(--accent)" }} />
          <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)" }}>
            Pipeline Orchestrator
          </span>
        </div>

        {/* Master Run Button */}
        <button
          onClick={onRunFullPipeline}
          disabled={isFullRunning}
          style={{
            width: "100%",
            padding: "12px 18px",
            marginBottom: 20,
            borderRadius: 12,
            background: isFullRunning
              ? "rgba(0,229,160,0.1)"
              : "linear-gradient(135deg, #00e5a0 0%, #00b8ff 100%)",
            color: isFullRunning ? "var(--accent)" : "#050608",
            fontWeight: 800,
            fontSize: 13,
            border: isFullRunning ? "1px solid rgba(0,229,160,0.3)" : "none",
            cursor: isFullRunning ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            boxShadow: isFullRunning ? "none" : "0 4px 20px rgba(0,229,160,0.3)",
            transition: "all 0.2s ease",
          }}
        >
          {isFullRunning ? (
            <>
              <ArrowClockwise size={15} style={{ animation: "spin 1s linear infinite" }} />
              <span>Pipeline Running ({fullStatus?.current_step || "init"})…</span>
            </>
          ) : (
            <>
              <Play size={15} weight="fill" />
              <span>Run End-to-End Pipeline</span>
            </>
          )}
        </button>

        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {steps.map((step) => {
            const s = statuses.find((x) => x.step === step);
            const done = s?.done ?? false;
            const isRunning = running === step || (isFullRunning && fullStatus?.current_step === step);
            return (
              <div
                key={step}
                style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "10px 14px",
                  background: done ? "rgba(0,229,160,0.04)" : "rgba(255,255,255,0.02)",
                  border: `1px solid ${done ? "rgba(0,229,160,0.15)" : "var(--border)"}`,
                  borderRadius: 10,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <StatusDot done={done} />
                  <span style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace", fontWeight: 500 }}>
                    {step}
                  </span>
                </div>
                <button
                  onClick={() => trigger(step)}
                  disabled={isRunning || isFullRunning}
                  style={{
                    padding: "4px 12px",
                    borderRadius: 6,
                    border: "1px solid var(--border)",
                    background: isRunning ? "rgba(0,229,160,0.1)" : "rgba(255,255,255,0.04)",
                    color: isRunning ? "var(--accent)" : "var(--muted)",
                    fontSize: 11,
                    fontWeight: 600,
                    cursor: (isRunning || isFullRunning) ? "not-allowed" : "pointer",
                    letterSpacing: "0.08em",
                    transition: "all 0.2s",
                  }}
                >
                  {isRunning ? "Running…" : "Run"}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  label, value, sub, accent,
}: {
  label: string; value: string | number; sub?: string; accent?: string;
}) {
  return (
    <motion.div
      className="stat-card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.32, 0.72, 0, 1] }}
    >
      <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)", marginBottom: 10 }}>
        {label}
      </div>
      <div style={{ fontSize: 36, fontWeight: 700, lineHeight: 1, color: accent ?? "var(--text)", letterSpacing: "-0.02em" }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>{sub}</div>
      )}
    </motion.div>
  );
}

function ClusterRow({
  cluster, index, onClick,
}: {
  cluster: ClusterSummary; index: number; onClick: () => void;
}) {
  const color = cluster.is_noise ? "var(--muted)" : clusterColor(cluster.cluster_id);
  const pct = Math.round((cluster.size / 200) * 100); // rough pct for bar

  return (
    <motion.div
      className="cluster-row"
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: index * 0.05, ease: [0.32, 0.72, 0, 1] }}
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        width: "100%",
        boxSizing: "border-box",
        padding: "14px 16px",
        overflow: "hidden",
      }}
    >
      {/* Accent bar */}
      <div className="accent-bar" style={{ height: 38, background: cluster.is_noise ? "var(--muted)" : `linear-gradient(180deg, ${color}, ${color}88)` }} />

      {/* ID badge */}
      <div style={{
        width: 34, height: 34, borderRadius: 10,
        background: cluster.is_noise ? "rgba(255,255,255,0.04)" : `${color}18`,
        border: `1px solid ${cluster.is_noise ? "var(--border)" : `${color}40`}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0, fontSize: 12, fontWeight: 700,
        fontFamily: "'JetBrains Mono', monospace",
        color: cluster.is_noise ? "var(--muted)" : color,
      }}>
        {cluster.is_noise ? "N" : cluster.cluster_id}
      </div>

      {/* Label + description */}
      <div style={{ flex: 1, minWidth: 0, overflow: "hidden" }}>
        <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {cluster.label ?? "(unlabelled)"}
        </div>
        <div style={{ fontSize: 11, color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {cluster.description ?? "—"}
        </div>
      </div>

      {/* Right metadata group */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0, marginLeft: "auto" }}>
        {/* Progress */}
        <div style={{ width: 70, flexShrink: 0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 3, color: "var(--muted)" }}>
            <span>{cluster.size} spans</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${Math.min(pct, 100)}%`, background: cluster.is_noise ? "var(--muted)" : `linear-gradient(90deg, ${color}, ${color}88)` }} />
          </div>
        </div>

        {/* Micro count */}
        {!cluster.is_noise && (
          <div className="badge blue" style={{ flexShrink: 0, padding: "2px 8px", fontSize: 10 }}>
            {cluster.micro_cluster_count} micro
          </div>
        )}

        <CaretRight size={14} style={{ color: "var(--muted)", flexShrink: 0 }} />
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Cluster Detail Modal
// ---------------------------------------------------------------------------

function ClusterModal({
  clusterId, onClose,
}: {
  clusterId: string; onClose: () => void;
}) {
  const [detail, setDetail] = useState<MacroClusterDetail | null>(null);
  const [spans, setSpans] = useState<SpanDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([fetchCluster(clusterId), fetchClusterSpans(clusterId, 30)])
      .then(([d, s]) => { setDetail(d); setSpans(s); })
      .catch((err) => {
        console.error("Failed to load cluster detail:", err);
        setError("Cluster details are temporarily unavailable or re-clustering is in progress.");
      })
      .finally(() => setLoading(false));
  }, [clusterId]);

  const color = clusterColor(clusterId);
  const micros = detail ? Object.entries(detail.micro_clusters).filter(([, m]) => !m.is_noise) : [];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, zIndex: 50,
        background: "rgba(5,6,8,0.85)", backdropFilter: "blur(12px)",
        display: "flex", alignItems: "center", justifyContent: "center",
        padding: "24px",
      }}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        transition={{ duration: 0.3, ease: [0.32, 0.72, 0, 1] }}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "100%", maxWidth: 800, maxHeight: "85dvh",
          background: "var(--card)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 24,
          overflow: "hidden",
          display: "flex", flexDirection: "column",
          boxShadow: `0 0 80px ${color}15`,
        }}
      >
        {/* Header */}
        <div style={{
          padding: "24px 28px 20px",
          borderBottom: "1px solid var(--border)",
          display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16,
        }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <div style={{
                width: 10, height: 10, borderRadius: "50%",
                background: color, boxShadow: `0 0 8px ${color}`,
              }} />
              <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "var(--muted)" }}>
                cluster/{clusterId}
              </span>
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: "-0.02em", marginBottom: 6 }}>
              {loading ? "Loading…" : (detail?.label ?? "(unlabelled)")}
            </div>
            {detail?.description && (
              <div style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.6 }}>{detail.description}</div>
            )}
          </div>
          <button
            onClick={onClose}
            style={{
              width: 32, height: 32, borderRadius: "50%",
              background: "rgba(255,255,255,0.06)", border: "1px solid var(--border)",
              display: "flex", alignItems: "center", justifyContent: "center",
              cursor: "pointer", color: "var(--muted)", flexShrink: 0,
            }}
          >
            <X size={14} />
          </button>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflow: "auto", padding: "20px 28px" }}>
          {loading ? (
            <div style={{ color: "var(--muted)", fontSize: 14, textAlign: "center", paddingTop: 40 }}>Loading spans…</div>
          ) : error ? (
            <div style={{ color: "#f87171", fontSize: 14, textAlign: "center", paddingTop: 40 }}>{error}</div>
          ) : (
            <>
              {/* Micro clusters */}
              {micros.length > 0 && (
                <div style={{ marginBottom: 24 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)", marginBottom: 12 }}>
                    Micro Clusters
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 10 }}>
                    {micros.map(([mid, m]) => (
                      <div key={mid} style={{
                        padding: "12px 14px", borderRadius: 12,
                        background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)",
                      }}>
                        <div style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "var(--muted)", marginBottom: 4 }}>
                          {clusterId}.{mid}
                        </div>
                        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{m.label ?? "(unlabelled)"}</div>
                        <div className="badge blue">{m.size} spans</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Spans */}
              <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)", marginBottom: 12 }}>
                Sample Spans ({spans.length})
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {spans.map((s) => (
                  <div key={s.span_id} className="span-row">
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                      <span style={{
                        fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
                        color: "var(--muted)",
                      }}>
                        {s.span_id.slice(0, 14)}…
                      </span>
                      <span className="tag">{s.operation_name || "unknown"}</span>
                      {s.model && <span className="tag">{s.model}</span>}
                      <span className={`tag status-${s.status.includes("ok") ? "ok" : s.status.includes("error") ? "error" : "unset"}`}>
                        {s.status}
                      </span>
                      {s.duration_ms != null && (
                        <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--muted)", fontFamily: "'JetBrains Mono', monospace" }}>
                          {Math.round(s.duration_ms)}ms
                        </span>
                      )}
                    </div>
                    {s.input_summary && (
                      <div style={{ fontSize: 11, color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        in: {s.input_summary.slice(0, 100)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Model distribution bar
// ---------------------------------------------------------------------------
function ModelBar({ dist }: { dist: Record<string, number> }) {
  const total = Object.values(dist).reduce((a, b) => a + b, 0);
  const entries = Object.entries(dist).slice(0, 6);
  const colors = ["var(--accent)", "var(--accent2)", "#a78bfa", "#f59e0b", "#f87171", "#34d399"];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {entries.map(([model, count], i) => (
        <div key={model}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
            <span style={{ color: "var(--text)", fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
              {model || "unknown"}
            </span>
            <span style={{ color: "var(--muted)" }}>{count}</span>
          </div>
          <div className="progress-track">
            <motion.div
              className="progress-fill"
              initial={{ width: 0 }}
              animate={{ width: `${(count / total) * 100}%` }}
              transition={{ duration: 0.8, delay: i * 0.1, ease: [0.32, 0.72, 0, 1] }}
              style={{ background: colors[i] }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Triage Card
// ---------------------------------------------------------------------------

const PROBLEM_COLORS: Record<string, string> = {
  api_error: "#f87171",
  exception_error: "#fb7185",
  tool_error: "#e879f9",
  high_latency: "#fb923c",
  truncated_output: "#fbbf24",
  empty_response: "#facc15",
  ml_anomaly: "#38bdf8",
  ml_confirmed: "#818cf8",
};
const PROBLEM_LABELS: Record<string, string> = {
  api_error: "API Error",
  exception_error: "Exception",
  tool_error: "Tool Error",
  high_latency: "High Latency",
  truncated_output: "Truncated Output",
  empty_response: "Empty Response",
  ml_anomaly: "ML Anomaly",
  ml_confirmed: "ML Confirmed",
};

function TriageCard({
  report,
  activeAlgo,
  switching,
  onSwitchAlgo,
}: {
  report: TriageReport;
  activeAlgo: "lof" | "ae";
  switching: boolean;
  onSwitchAlgo: (a: "lof" | "ae") => void;
}) {
  const algoData = activeAlgo === "lof" ? report.lof : report.ae;
  const total = report.total_spans;
  const probFrac = total > 0 ? (algoData.problematic / total) * 100 : 0;
  const typeEntries = Object.entries(algoData.problem_types)
    .filter(([type]) => type in PROBLEM_COLORS)
    .slice(0, 8);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.32, 0.72, 0, 1] }}
      style={{ marginBottom: 40 }}
    >
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{ width: 3, height: 16, borderRadius: 2, background: "linear-gradient(180deg, #f87171, #fb923c)" }} />
        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)" }}>
          Triage Analysis
        </span>
      </div>

      <div style={{
        background: "linear-gradient(135deg, rgba(17,19,24,0.85) 0%, rgba(13,15,19,0.95) 100%)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 20,
        padding: "20px 24px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.3)",
      }}>
        {/* Top row: algo toggle + summary stats */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
          {/* Algorithm toggle */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase" }}>
              Algorithm
            </span>
            <div style={{ display: "flex", gap: 4, background: "rgba(255,255,255,0.04)", borderRadius: 999, padding: 3, border: "1px solid var(--border)" }}>
              {(["lof", "ae"] as const).map((algo) => (
                <button
                  key={algo}
                  id={`triage-algo-${algo}`}
                  onClick={() => onSwitchAlgo(algo)}
                  disabled={switching}
                  style={{
                    padding: "5px 16px",
                    borderRadius: 999,
                    fontSize: 12,
                    fontWeight: 700,
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                    background: activeAlgo === algo
                      ? "linear-gradient(135deg, #f87171, #fb923c)"
                      : "transparent",
                    color: activeAlgo === algo ? "#fff" : "var(--muted)",
                    border: "none",
                    cursor: switching ? "not-allowed" : "pointer",
                    opacity: switching && activeAlgo !== algo ? 0.5 : 1,
                    transition: "all 0.2s ease",
                    boxShadow: activeAlgo === algo ? "0 2px 12px rgba(248,113,113,0.4)" : "none",
                  }}
                >
                  {algo === "lof" ? "LOF" : "Autoencoder"}
                </button>
              ))}
            </div>
            {switching && (
              <span style={{ fontSize: 11, color: "var(--muted)", animation: "pulse 1s infinite" }}>Applying…</span>
            )}
          </div>

          {/* Summary chips */}
          <div style={{ display: "flex", gap: 10 }}>
            <div style={{
              padding: "6px 14px", borderRadius: 10,
              background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)",
            }}>
              <span style={{ fontSize: 10, color: "#f87171", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>Problematic</span>
              <div style={{ fontSize: 22, fontWeight: 800, color: "#f87171", lineHeight: 1.1 }}>{algoData.problematic}</div>
            </div>
            <div style={{
              padding: "6px 14px", borderRadius: 10,
              background: "rgba(0,229,160,0.06)", border: "1px solid rgba(0,229,160,0.15)",
            }}>
              <span style={{ fontSize: 10, color: "var(--accent)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>Healthy</span>
              <div style={{ fontSize: 22, fontWeight: 800, color: "var(--accent)", lineHeight: 1.1 }}>{algoData.healthy}</div>
            </div>
            <div style={{
              padding: "6px 14px", borderRadius: 10,
              background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)",
            }}>
              <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>Total</span>
              <div style={{ fontSize: 22, fontWeight: 800, color: "var(--text)", lineHeight: 1.1 }}>{total}</div>
            </div>
          </div>
        </div>

        {/* Problematic fraction bar */}
        <div style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--muted)", marginBottom: 5 }}>
            <span>Problematic fraction</span>
            <span style={{ color: "#f87171", fontWeight: 600 }}>{probFrac.toFixed(1)}%</span>
          </div>
          <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 999, overflow: "hidden" }}>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${probFrac}%` }}
              transition={{ duration: 0.8, ease: [0.32, 0.72, 0, 1] }}
              style={{
                height: "100%",
                background: "linear-gradient(90deg, #f87171, #fb923c)",
                borderRadius: 999,
              }}
            />
          </div>
        </div>

        {/* Problem type breakdown */}
        {typeEntries.length > 0 && (
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>
              Problem Type Breakdown
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
              {typeEntries.map(([type, count]) => {
                const color = PROBLEM_COLORS[type] ?? "#888";
                const label = PROBLEM_LABELS[type] ?? type;
                const barPct = algoData.problematic > 0 ? (count / algoData.problematic) * 100 : 0;
                return (
                  <div key={type}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: `0 0 4px ${color}` }} />
                        <span style={{ color: "var(--text)" }}>{label}</span>
                      </div>
                      <span style={{ color, fontWeight: 600 }}>{count}</span>
                    </div>
                    <div style={{ height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 999, overflow: "hidden" }}>
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${barPct}%` }}
                        transition={{ duration: 0.6, ease: [0.32, 0.72, 0, 1] }}
                        style={{ height: "100%", background: color, borderRadius: 999 }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Latency threshold note */}
        <div style={{ marginTop: 14, fontSize: 10, color: "var(--muted)", borderTop: "1px solid var(--border)", paddingTop: 10 }}>
          P95 latency threshold: <span style={{ color: "#fb923c", fontWeight: 600 }}>{report.p95_latency_threshold_ms.toFixed(0)} ms</span>
          &nbsp;·&nbsp; Rules fired on <span style={{ color: "var(--text)" }}>{report.rules.fired}</span> spans
        </div>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Dashboard() {
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [statuses, setStatuses] = useState<PipelineStatus[]>([]);
  const [points, setPoints] = useState<ClusterPoint[]>([]);
  const [triageReport, setTriageReport] = useState<TriageReport | null>(null);
  const [activeAlgo, setActiveAlgo] = useState<"lof" | "ae">("lof");
  const [algoSwitching, setAlgoSwitching] = useState(false);
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [activeTab, setActiveTab] = useState<"Clusters" | "Pipeline" | "Stats">("Clusters");

  // End-to-End Pipeline state
  const [fullStatus, setFullStatus] = useState<FullPipelineStatus | null>(null);
  const [isFullRunning, setIsFullRunning] = useState(false);
  const [showProgressBanner, setShowProgressBanner] = useState(false);

  const load = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const [cl, st, ss, pts] = await Promise.all([
        fetchClusters(),
        fetchStats(),
        fetchStatus(),
        fetchClusterPoints().catch(() => []),
      ]);
      setClusters(cl);
      setStats(st);
      setStatuses(ss);
      setPoints(pts);
      fetchTriageReport().then(setTriageReport).catch(() => { });
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to connect to API. Is the backend running?");
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  const startFullPipelinePolling = useCallback(() => {
    setIsFullRunning(true);
    setShowProgressBanner(true);

    const interval = setInterval(async () => {
      try {
        const st = await fetchFullPipelineStatus();
        setFullStatus(st);
        if (!st.is_running) {
          clearInterval(interval);
          setIsFullRunning(false);
          await load();
        }
      } catch (err) {
        console.error("Error fetching full pipeline status:", err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [load]);

  const handleRunFullPipeline = useCallback(async () => {
    if (isFullRunning) return;
    try {
      await runFullPipeline();
      startFullPipelinePolling();
    } catch (err) {
      console.error("Failed to start full pipeline:", err);
      setError("Failed to start end-to-end pipeline.");
    }
  }, [isFullRunning, startFullPipelinePolling]);

  useEffect(() => {
    fetchFullPipelineStatus().then((st) => {
      if (st.is_running) {
        setFullStatus(st);
        startFullPipelinePolling();
      }
    }).catch(() => { });
  }, [startFullPipelinePolling]);

  // Switch active triage algorithm
  const handleAlgoSwitch = useCallback(async (algo: "lof" | "ae") => {
    if (algo === activeAlgo || algoSwitching) return;
    setAlgoSwitching(true);
    try {
      await selectTriageAlgorithm(algo);
      setActiveAlgo(algo);
      setTimeout(async () => {
        await load();
        setAlgoSwitching(false);
      }, 3000);
    } catch (err) {
      console.error("Failed to switch algorithm:", err);
      setAlgoSwitching(false);
    }
  }, [activeAlgo, algoSwitching, load]);

  useEffect(() => { load(); }, [load]);

  const scrollToSection = (tab: "Clusters" | "Pipeline" | "Stats") => {
    setActiveTab(tab);
    const idMap = {
      Clusters: "section-clusters",
      Pipeline: "section-pipeline",
      Stats: "section-stats",
    };
    const target = document.getElementById(idMap[tab]);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const filtered = clusters.filter((c) =>
    !search || (c.label ?? "").toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{ position: "relative", zIndex: 1, minHeight: "100dvh" }}>
      {/* ── Top Nav Header ── */}
      <header style={{
        position: "sticky", top: 0, zIndex: 50,
        width: "100%", backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)",
        background: "rgba(5,6,8,0.85)", borderBottom: "1px solid var(--border)",
        padding: "12px 24px", marginBottom: 36,
        display: "flex", justifyContent: "center", alignItems: "center",
      }}>
        <div className="nav-pill" style={{
          display: "flex", alignItems: "center", gap: 0,
          padding: "6px 18px",
        }}>
          <Graph size={18} weight="duotone" style={{ color: "var(--accent)", marginRight: 10 }} />
          <span style={{ fontWeight: 700, fontSize: 14, letterSpacing: "-0.01em", marginRight: 20 }}>
            Cluster Observatory
          </span>
          <div style={{ display: "flex", gap: 4 }}>
            {(["Clusters", "Pipeline", "Stats"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => scrollToSection(tab)}
                style={{
                  fontSize: 12, fontWeight: 500, padding: "4px 14px", borderRadius: 999,
                  color: activeTab === tab ? "var(--accent)" : "var(--muted)",
                  background: activeTab === tab ? "rgba(0,229,160,0.12)" : "transparent",
                  border: activeTab === tab ? "1px solid rgba(0,229,160,0.3)" : "1px solid transparent",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Master Run Pipeline Header Action Button */}
          <button
            onClick={handleRunFullPipeline}
            disabled={isFullRunning}
            style={{
              marginLeft: 16, padding: "5px 16px", borderRadius: 999,
              background: isFullRunning
                ? "rgba(0,229,160,0.1)"
                : "linear-gradient(135deg, #00e5a0 0%, #00b8ff 100%)",
              border: isFullRunning ? "1px solid rgba(0,229,160,0.3)" : "none",
              color: isFullRunning ? "var(--accent)" : "#050608",
              fontSize: 12, fontWeight: 700,
              cursor: isFullRunning ? "not-allowed" : "pointer",
              display: "flex", alignItems: "center", gap: 6,
              boxShadow: isFullRunning ? "none" : "0 2px 14px rgba(0,229,160,0.3)",
              transition: "all 0.15s ease",
            }}
          >
            {isFullRunning ? (
              <>
                <ArrowClockwise size={13} style={{ animation: "spin 1s linear infinite" }} />
                <span>Running Pipeline…</span>
              </>
            ) : (
              <>
                <Play size={13} weight="fill" />
                <span>Run Pipeline</span>
              </>
            )}
          </button>

          <button
            onClick={load}
            disabled={isRefreshing}
            style={{
              marginLeft: 10, padding: "5px 14px", borderRadius: 999,
              background: "rgba(255,255,255,0.05)", border: "1px solid var(--border)",
              color: isRefreshing ? "var(--accent)" : "var(--muted)", fontSize: 12, fontWeight: 500,
              cursor: isRefreshing ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: 6,
              transition: "all 0.15s ease",
            }}
          >
            <ArrowClockwise
              size={13}
              style={{
                animation: isRefreshing ? "spin 1s linear infinite" : "none",
              }}
            />
            {isRefreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "0 24px 80px" }}>

        {/* ── Progress Banner ── */}
        <AnimatePresence>
          {showProgressBanner && fullStatus && (
            <PipelineProgressBanner
              status={fullStatus}
              onClose={() => setShowProgressBanner(false)}
              onRunAgain={handleRunFullPipeline}
            />
          )}
        </AnimatePresence>

        {/* ── Error banner ── */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
            style={{
              marginBottom: 24, padding: "14px 20px", borderRadius: 12,
              background: "rgba(255,107,107,0.08)", border: "1px solid rgba(255,107,107,0.2)",
              display: "flex", alignItems: "center", gap: 10, fontSize: 13,
            }}
          >
            <WarningCircle size={16} style={{ color: "#ff6b6b", flexShrink: 0 }} />
            <span style={{ color: "#ff6b6b" }}>{error}</span>
          </motion.div>
        )}

        {/* ── Hero header ── */}
        <div style={{ marginBottom: 40 }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.32, 0.72, 0, 1] }}
          >
            <div className="badge accent" style={{ marginBottom: 16 }}>
              <Graph size={10} weight="fill" />
              Observability Intelligence
            </div>
            <h1 style={{
              fontSize: "clamp(36px, 5vw, 64px)", fontWeight: 800,
              letterSpacing: "-0.04em", lineHeight: 1,
              background: "linear-gradient(135deg, var(--text) 40%, var(--muted) 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              marginBottom: 16,
            }}>
              Span Cluster<br />Analysis
            </h1>
            <p style={{ fontSize: 16, color: "var(--muted)", maxWidth: 480, lineHeight: 1.7 }}>
              Two-level embedding clustering on Mistral Observability spans — macro categories surfaced by UMAP + HDBSCAN, auto-labelled by mistral-large.
            </p>
          </motion.div>
        </div>

        {/* ── Stats row ── */}
        {stats && (
          <div
            id="section-stats"
            style={{
              display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
              gap: 14, marginBottom: 40, scrollMarginTop: 100,
            }}
          >
            <StatCard label="Total Spans" value={stats.total_spans} sub="fetched from API" accent="var(--accent)" />
            <StatCard label="Macro Clusters" value={stats.macro_clusters} sub="top-level groups" />
            <StatCard label="Micro Clusters" value={stats.micro_clusters} sub="sub-categories" accent="var(--accent2)" />
            <StatCard label="Noise Spans" value={stats.noise_spans} sub="unclustered" />
            <StatCard label="Labelled" value={`${stats.labelled}/${stats.macro_clusters}`} sub="by mistral-large" accent="#a78bfa" />
          </div>
        )}

        {/* ── Triage Card (shown when triage report available) ── */}
        {triageReport && (
          <TriageCard
            report={triageReport}
            activeAlgo={activeAlgo}
            switching={algoSwitching}
            onSwitchAlgo={handleAlgoSwitch}
          />
        )}

        {/* ── Cluster charts ── */}
        {!loading && clusters.length > 0 && (
          <ClusterCharts clusters={clusters} />
        )}

        {/* ── 2D Cluster Embedding Map Scatter Chart ── */}
        {!loading && points.length > 0 && (
          <div style={{ marginBottom: 40 }}>
            <ClusterScatterPlot
              points={points}
              clusters={clusters}
              onSelectCluster={(cid) => setSelected(cid)}
            />
          </div>
        )}

        {/* ── Main grid ── */}
        <div className="main-dashboard-grid" style={{ display: "grid", gap: 20, alignItems: "start" }}>

          {/* Left: Clusters list */}
          <div id="section-clusters" style={{ scrollMarginTop: 100 }}>
            {/* Search */}
            <div style={{
              display: "flex", alignItems: "center", gap: 10,
              padding: "10px 16px", borderRadius: 12, marginBottom: 16,
              background: "var(--card)", border: "1px solid var(--border)",
            }}>
              <MagnifyingGlass size={14} style={{ color: "var(--muted)", flexShrink: 0 }} />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search clusters…"
                style={{
                  flex: 1, background: "none", border: "none", outline: "none",
                  color: "var(--text)", fontSize: 13, fontFamily: "inherit",
                }}
              />
            </div>

            {/* Section label */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
              <Rows size={14} style={{ color: "var(--muted)" }} />
              <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)" }}>
                Macro Clusters ({filtered.length})
              </span>
            </div>

            {loading ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {[...Array(5)].map((_, i) => (
                  <div key={i} style={{
                    height: 72, borderRadius: 14, background: "var(--card)",
                    border: "1px solid var(--border)", opacity: 0.5,
                  }} />
                ))}
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {filtered.map((c, i) => (
                  <ClusterRow
                    key={c.cluster_id}
                    cluster={c}
                    index={i}
                    onClick={() => setSelected(c.cluster_id)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Right sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Pipeline */}
            <div id="section-pipeline" style={{ scrollMarginTop: 100 }}>
              <PipelinePanel
                statuses={statuses}
                onRefresh={load}
                onRunFullPipeline={handleRunFullPipeline}
                isFullRunning={isFullRunning}
                fullStatus={fullStatus}
              />
            </div>

            {/* Model distribution */}
            {stats?.model_distribution && Object.keys(stats.model_distribution).length > 0 && (
              <div className="bezel">
                <div className="bezel-inner" style={{ padding: "20px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                    <Cpu size={14} style={{ color: "var(--accent2)" }} />
                    <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)" }}>
                      Models
                    </span>
                  </div>
                  <ModelBar dist={stats.model_distribution} />
                </div>
              </div>
            )}

            {/* Operations */}
            {stats?.operation_distribution && Object.keys(stats.operation_distribution).length > 0 && (
              <div className="bezel">
                <div className="bezel-inner" style={{ padding: "20px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                    <ChartBar size={14} style={{ color: "#a78bfa" }} />
                    <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)" }}>
                      Operations
                    </span>
                  </div>
                  <ModelBar dist={stats.operation_distribution} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Cluster detail modal ── */}
      <AnimatePresence>
        {selected && (
          <ClusterModal clusterId={selected} onClose={() => setSelected(null)} />
        )}
      </AnimatePresence>
    </div>
  );
}
