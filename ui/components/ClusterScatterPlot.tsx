"use client";

import { useState, useMemo } from "react";
import { motion } from "motion/react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { type ClusterPoint, type ClusterSummary } from "@/lib/api";

// ---------------------------------------------------------------------------
// Problem-type colour palette
// ---------------------------------------------------------------------------
const PROBLEM_COLORS: Record<string, string> = {
  api_error:        "#f87171",   // red
  exception_error:  "#fb7185",   // rose
  tool_error:       "#e879f9",   // fuchsia
  high_latency:     "#fb923c",   // orange
  truncated_output: "#fbbf24",   // amber
  empty_response:   "#facc15",   // yellow
  ml_anomaly:       "#38bdf8",   // sky blue
  ml_confirmed:     "#818cf8",   // indigo
};

const PROBLEM_LABELS: Record<string, string> = {
  api_error:        "API Error",
  exception_error:  "Exception",
  tool_error:       "Tool Error",
  high_latency:     "High Latency",
  truncated_output: "Truncated",
  empty_response:   "Empty Response",
  ml_anomaly:       "ML Anomaly",
  ml_confirmed:     "ML Confirmed",
};

const CLUSTER_COLORS = [
  "#00e5a0", "#00b8ff", "#a78bfa", "#f59e0b",
  "#f87171", "#34d399", "#60a5fa", "#fb923c",
];

function clusterColor(id: string, isNoise = false): string {
  if (isNoise || id === "-1") return "#5a6070";
  const n = parseInt(id, 10);
  return CLUSTER_COLORS[Math.abs(n) % CLUSTER_COLORS.length];
}

function problemTypeColor(types: string[] | undefined): string {
  if (!types || types.length === 0) return "#5a6070";
  // Priority order: most severe first
  const priority = [
    "api_error", "tool_error", "exception_error",
    "high_latency", "truncated_output", "empty_response",
    "ml_anomaly", "ml_confirmed",
  ];
  for (const p of priority) {
    if (types.includes(p)) return PROBLEM_COLORS[p] ?? "#00e5a0";
  }
  const cleanType = types.find((t) => t in PROBLEM_COLORS);
  return cleanType ? PROBLEM_COLORS[cleanType] : "#5a6070";
}

interface ClusterScatterPlotProps {
  points: ClusterPoint[];
  clusters: ClusterSummary[];
  onSelectCluster?: (clusterId: string) => void;
  colorMode?: "cluster" | "problem";
}

export function ClusterScatterPlot({
  points,
  clusters,
  onSelectCluster,
  colorMode: initialColorMode = "problem",
}: ClusterScatterPlotProps) {
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [colorMode, setColorMode] = useState<"cluster" | "problem">(initialColorMode);

  const filteredPoints = useMemo(() => {
    if (!selectedClusterId) return points;
    return points.filter((p) => p.cluster_id === selectedClusterId);
  }, [points, selectedClusterId]);

  // Collect unique problem types present in data
  const presentProblemTypes = useMemo(() => {
    const types = new Set<string>();
    points.forEach((p) =>
      (p.problem_types ?? []).forEach((t) => {
        if (t in PROBLEM_COLORS) {
          types.add(t);
        }
      })
    );
    return Array.from(types);
  }, [points]);

  if (!points || points.length === 0) {
    return (
      <div
        style={{
          background: "linear-gradient(135deg, rgba(17,19,24,0.7) 0%, rgba(13,15,19,0.9) 100%)",
          border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: 16,
          padding: 24,
          textAlign: "center",
          color: "#5a6070",
        }}
      >
        No 2D cluster points available. Run the embedding &amp; clustering pipeline step first.
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      style={{
        background: "linear-gradient(135deg, rgba(17,19,24,0.85) 0%, rgba(13,15,19,0.95) 100%)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 20,
        padding: "22px 24px",
        boxShadow: "0 16px 40px rgba(0,0,0,0.4)",
        position: "relative",
      }}
    >
      {/* Header & Controls */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: 16,
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 14,
              fontWeight: 600,
              color: "#e2e8f0",
              letterSpacing: "-0.01em",
            }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#f87171",
                boxShadow: "0 0 10px #f87171",
              }}
            />
            Problematic Span Cluster Map
          </div>
          <p style={{ fontSize: 12, color: "#64748b", marginTop: 4, margin: 0 }}>
            {points.length} problematic span{points.length !== 1 ? "s" : ""} projected via UMAP · hover for details
          </p>
        </div>

        {/* Color mode + cluster filter */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-end" }}>
          {/* Color mode toggle */}
          <div style={{ display: "flex", gap: 4 }}>
            {(["problem", "cluster"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setColorMode(mode)}
                style={{
                  padding: "3px 10px",
                  borderRadius: 20,
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  border: colorMode === mode ? "1px solid #00e5a0" : "1px solid rgba(255,255,255,0.1)",
                  background: colorMode === mode ? "rgba(0,229,160,0.12)" : "rgba(255,255,255,0.03)",
                  color: colorMode === mode ? "#00e5a0" : "#64748b",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {mode === "problem" ? "By Problem Type" : "By Cluster"}
              </button>
            ))}
          </div>

          {/* Cluster filter pills */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, maxWidth: 480, justifyContent: "flex-end" }}>
            <button
              onClick={() => setSelectedClusterId(null)}
              style={{
                padding: "3px 9px",
                borderRadius: 20,
                fontSize: 10,
                fontWeight: 500,
                border: selectedClusterId === null ? "1px solid #00e5a0" : "1px solid rgba(255,255,255,0.1)",
                background: selectedClusterId === null ? "rgba(0,229,160,0.15)" : "rgba(255,255,255,0.03)",
                color: selectedClusterId === null ? "#00e5a0" : "#94a3b8",
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              All ({points.length})
            </button>
            {clusters.map((c) => {
              const color = clusterColor(c.cluster_id, c.is_noise);
              const isSelected = selectedClusterId === c.cluster_id;
              return (
                <button
                  key={c.cluster_id}
                  onClick={() => {
                    const newId = isSelected ? null : c.cluster_id;
                    setSelectedClusterId(newId);
                    if (onSelectCluster && newId) onSelectCluster(newId);
                  }}
                  style={{
                    padding: "3px 9px",
                    borderRadius: 20,
                    fontSize: 10,
                    fontWeight: 500,
                    border: isSelected ? `1px solid ${color}` : "1px solid rgba(255,255,255,0.08)",
                    background: isSelected ? `${color}22` : "rgba(255,255,255,0.03)",
                    color: isSelected ? color : "#94a3b8",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    transition: "all 0.15s ease",
                  }}
                >
                  <span style={{ width: 5, height: 5, borderRadius: "50%", background: color }} />
                  {c.label || `C${c.cluster_id}`} ({c.size})
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Problem-type legend (shown when colorMode === "problem") */}
      {colorMode === "problem" && presentProblemTypes.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "5px 14px", marginBottom: 12 }}>
          {presentProblemTypes.map((pt) => (
            <div key={pt} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <div style={{
                width: 7, height: 7, borderRadius: "50%",
                background: PROBLEM_COLORS[pt] ?? "#888",
                boxShadow: `0 0 5px ${PROBLEM_COLORS[pt] ?? "#888"}`,
              }} />
              <span style={{ fontSize: 10, color: "#94a3b8" }}>
                {PROBLEM_LABELS[pt] ?? pt}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Scatter Chart */}
      <div style={{ width: "100%", height: 340, position: "relative" }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
            <XAxis
              type="number"
              dataKey="x"
              domain={[-110, 110]}
              tick={{ fill: "#475569", fontSize: 10 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
              tickLine={false}
              name="X Projection"
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[-110, 110]}
              tick={{ fill: "#475569", fontSize: 10 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
              tickLine={false}
              name="Y Projection"
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const pt = payload[0].payload as ClusterPoint;
                const dotColor = colorMode === "problem"
                  ? problemTypeColor(pt.problem_types)
                  : clusterColor(pt.cluster_id, pt.is_noise);
                const primaryType = (pt.problem_types ?? [])[0];

                return (
                  <div
                    style={{
                      background: "#0f172a",
                      border: `1px solid ${dotColor}44`,
                      borderRadius: 12,
                      padding: "12px 16px",
                      fontSize: 12,
                      boxShadow: "0 12px 32px rgba(0,0,0,0.7)",
                      maxWidth: 300,
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, color: dotColor, marginBottom: 6 }}>
                      <span style={{ width: 8, height: 8, borderRadius: "50%", background: dotColor }} />
                      {pt.cluster_label}
                    </div>

                    <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 6 }}>
                      Span: <code style={{ color: "#e2e8f0" }}>{pt.span_id.slice(0, 14)}…</code>
                    </div>

                    {/* Problem types badges */}
                    {(pt.problem_types ?? []).filter((t) => t in PROBLEM_COLORS).length > 0 && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 8 }}>
                        {(pt.problem_types ?? []).filter((t) => t in PROBLEM_COLORS).map((t) => (
                          <span
                            key={t}
                            style={{
                              padding: "2px 7px", borderRadius: 20, fontSize: 10, fontWeight: 600,
                              background: `${PROBLEM_COLORS[t] ?? "#888"}22`,
                              border: `1px solid ${PROBLEM_COLORS[t] ?? "#888"}44`,
                              color: PROBLEM_COLORS[t] ?? "#888",
                            }}
                          >
                            {PROBLEM_LABELS[t] ?? t}
                          </span>
                        ))}
                      </div>
                    )}

                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr",
                        gap: 5,
                        fontSize: 11,
                        background: "rgba(255,255,255,0.03)",
                        padding: "6px 8px",
                        borderRadius: 6,
                        marginBottom: 6,
                      }}
                    >
                      <div><span style={{ color: "#64748b" }}>Op: </span><span style={{ color: "#cbd5e1" }}>{pt.operation_name || "N/A"}</span></div>
                      <div><span style={{ color: "#64748b" }}>Model: </span><span style={{ color: "#cbd5e1" }}>{pt.model || "N/A"}</span></div>
                      <div>
                        <span style={{ color: "#64748b" }}>Status: </span>
                        <span style={{ color: pt.status === "ok" ? "#00e5a0" : "#f87171" }}>{pt.status || "OK"}</span>
                      </div>
                      <div>
                        <span style={{ color: "#64748b" }}>Latency: </span>
                        <span style={{ color: "#cbd5e1" }}>{pt.duration_ms ? `${pt.duration_ms.toFixed(1)} ms` : "N/A"}</span>
                      </div>
                    </div>

                    {/* LOF / AE scores */}
                    {(pt.lof_score != null || pt.ae_recon_error != null) && (
                      <div style={{ display: "flex", gap: 8, fontSize: 10 }}>
                        {pt.lof_score != null && (
                          <div style={{
                            padding: "2px 7px", borderRadius: 6,
                            background: "rgba(56,189,248,0.1)", color: "#38bdf8", border: "1px solid #38bdf844",
                          }}>
                            LOF {pt.lof_score.toFixed(3)}
                          </div>
                        )}
                        {pt.ae_recon_error != null && (
                          <div style={{
                            padding: "2px 7px", borderRadius: 6,
                            background: "rgba(129,140,248,0.1)", color: "#818cf8", border: "1px solid #818cf844",
                          }}>
                            AE {pt.ae_recon_error.toFixed(5)}
                          </div>
                        )}
                      </div>
                    )}

                    {pt.input_summary && (
                      <div style={{ fontSize: 10, color: "#64748b", fontStyle: "italic", marginTop: 6 }}>
                        &ldquo;{pt.input_summary}&rdquo;
                      </div>
                    )}
                  </div>
                );
              }}
            />
            <Scatter name="Spans" data={filteredPoints}>
              {filteredPoints.map((entry, index) => {
                const dotColor = colorMode === "problem"
                  ? problemTypeColor(entry.problem_types)
                  : clusterColor(entry.cluster_id, entry.is_noise);
                const isSelected = selectedClusterId === entry.cluster_id;
                const radius = isSelected ? 7 : entry.is_noise ? 3.5 : 5;
                const opacity = selectedClusterId && !isSelected ? 0.2 : 0.88;

                return (
                  <Cell
                    key={`cell-${index}`}
                    fill={dotColor}
                    stroke={isSelected ? "#ffffff" : dotColor}
                    strokeWidth={isSelected ? 1.5 : 0}
                    r={radius}
                    fillOpacity={opacity}
                  />
                );
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

