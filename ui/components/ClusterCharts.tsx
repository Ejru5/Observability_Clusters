"use client";

import { useMemo } from "react";
import { motion } from "motion/react";
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  RadialBarChart, RadialBar, Legend,
} from "recharts";
import { type ClusterSummary } from "@/lib/api";

// ---------------------------------------------------------------------------
// Colour palette (must match page.tsx)
// ---------------------------------------------------------------------------
const CLUSTER_COLORS = [
  "#00e5a0", "#00b8ff", "#a78bfa", "#f59e0b",
  "#f87171", "#34d399", "#60a5fa", "#fb923c",
];
function clusterColor(id: string, fallback = "#5a6070"): string {
  if (id === "-1") return fallback;
  const n = parseInt(id, 10);
  return CLUSTER_COLORS[Math.abs(n) % CLUSTER_COLORS.length];
}

// ---------------------------------------------------------------------------
// Shared tooltip
// ---------------------------------------------------------------------------
function DarkTooltip({ active, payload }: { active?: boolean; payload?: { name: string; value: number; payload: { color?: string } }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0];
  return (
    <div style={{
      background: "#111318", border: "1px solid rgba(255,255,255,0.1)",
      borderRadius: 10, padding: "10px 14px", fontSize: 12,
      boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    }}>
      <div style={{ fontWeight: 600, color: p.payload.color ?? "#fff", marginBottom: 2 }}>{p.name}</div>
      <div style={{ color: "#5a6070" }}>{p.value} spans</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 1. Donut chart — span share per cluster
// ---------------------------------------------------------------------------
function DonutChart({ clusters }: { clusters: ClusterSummary[] }) {
  const data = clusters
    .filter((c) => !c.is_noise)
    .map((c) => ({
      name: c.label ?? `Cluster ${c.cluster_id}`,
      value: c.size,
      color: clusterColor(c.cluster_id),
    }));

  const total = data.reduce((s, d) => s + d.value, 0);

  return (
    <div style={{ position: "relative" }}>
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={65}
            outerRadius={95}
            paddingAngle={3}
            dataKey="value"
            strokeWidth={0}
          >
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} opacity={0.9} />
            ))}
          </Pie>
          <Tooltip content={<DarkTooltip />} />
        </PieChart>
      </ResponsiveContainer>
      {/* Centre label */}
      <div style={{
        position: "absolute", top: "50%", left: "50%",
        transform: "translate(-50%, -50%)",
        textAlign: "center", pointerEvents: "none",
      }}>
        <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.04em", color: "var(--text)" }}>
          {total}
        </div>
        <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase" }}>
          spans
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 2. Bar chart — spans per cluster
// ---------------------------------------------------------------------------
function ClusterBarChart({ clusters }: { clusters: ClusterSummary[] }) {
  const data = clusters
    .filter((c) => !c.is_noise)
    .map((c) => ({
      name: c.label ? c.label.split(" ").slice(0, 2).join(" ") : `C${c.cluster_id}`,
      fullName: c.label ?? `Cluster ${c.cluster_id}`,
      spans: c.size,
      micro: c.micro_cluster_count,
      color: clusterColor(c.cluster_id),
    }))
    .sort((a, b) => b.spans - a.spans);

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }} barSize={18}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fill: "#5a6070", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          interval={0}
          angle={-20}
          textAnchor="end"
          height={42}
        />
        <YAxis
          tick={{ fill: "#5a6070", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.04)", radius: 6 }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload;
            return (
              <div style={{
                background: "#111318", border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 10, padding: "10px 14px", fontSize: 12,
              }}>
                <div style={{ fontWeight: 600, color: d.color, marginBottom: 4 }}>{d.fullName}</div>
                <div style={{ color: "#5a6070" }}>{d.spans} spans · {d.micro} micro-clusters</div>
              </div>
            );
          }}
        />
        <Bar dataKey="spans" radius={[6, 6, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.color} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// 3. Radial bar — micro-cluster depth per cluster
// ---------------------------------------------------------------------------
function RadialDepthChart({ clusters }: { clusters: ClusterSummary[] }) {
  const data = clusters
    .filter((c) => !c.is_noise && c.micro_cluster_count > 0)
    .map((c, i) => ({
      name: c.label ? c.label.split(" ").slice(0, 2).join(" ") : `C${c.cluster_id}`,
      value: c.micro_cluster_count,
      fill: CLUSTER_COLORS[i % CLUSTER_COLORS.length],
    }));

  if (!data.length) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 220, color: "var(--muted)", fontSize: 13 }}>
      No micro-clusters yet
    </div>
  );

  return (
    <ResponsiveContainer width="100%" height={220}>
      <RadialBarChart
        innerRadius={20}
        outerRadius={90}
        data={data}
        startAngle={90}
        endAngle={-270}
      >
        <RadialBar
          dataKey="value"
          cornerRadius={6}
          background={{ fill: "rgba(255,255,255,0.03)" }}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload;
            return (
              <div style={{
                background: "#111318", border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 10, padding: "10px 14px", fontSize: 12,
              }}>
                <div style={{ fontWeight: 600, color: d.fill, marginBottom: 2 }}>{d.name}</div>
                <div style={{ color: "#5a6070" }}>{d.value} micro-clusters</div>
              </div>
            );
          }}
        />
        <Legend
          iconSize={8}
          iconType="circle"
          wrapperStyle={{ fontSize: 10, color: "#5a6070", paddingTop: 8 }}
        />
      </RadialBarChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Legend strip below donut
// ---------------------------------------------------------------------------
function DonutLegend({ clusters }: { clusters: ClusterSummary[] }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px", justifyContent: "center", marginTop: 4 }}>
      {clusters.filter((c) => !c.is_noise).map((c) => (
        <div key={c.cluster_id} style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <div style={{
            width: 7, height: 7, borderRadius: "50%",
            background: clusterColor(c.cluster_id), flexShrink: 0,
          }} />
          <span style={{ fontSize: 10, color: "var(--muted)", maxWidth: 90, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {c.label?.split(" ").slice(0, 2).join(" ") ?? `C${c.cluster_id}`}
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main export — the full charts section
// ---------------------------------------------------------------------------
export function ClusterCharts({ clusters }: { clusters: ClusterSummary[] }) {
  const hasData = clusters.filter((c) => !c.is_noise).length > 0;

  if (!hasData) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.2, ease: [0.32, 0.72, 0, 1] }}
      style={{ marginBottom: 40 }}
    >
      {/* Section header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{ width: 3, height: 16, borderRadius: 2, background: "linear-gradient(180deg, var(--accent), var(--accent2))" }} />
        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)" }}>
          Cluster Visualisation
        </span>
      </div>

      {/* 3-column responsive chart grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 320px), 1fr))",
        gap: 16,
      }}>

        {/* Card 1 — Donut */}
        <div className="bezel">
          <div className="bezel-inner" style={{ padding: "20px" }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--muted)", marginBottom: 16, letterSpacing: "0.06em" }}>
              Span Share
            </div>
            <DonutChart clusters={clusters} />
            <DonutLegend clusters={clusters} />
          </div>
        </div>

        {/* Card 2 — Bar */}
        <div className="bezel">
          <div className="bezel-inner" style={{ padding: "20px" }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--muted)", marginBottom: 16, letterSpacing: "0.06em" }}>
              Spans per Cluster
            </div>
            <ClusterBarChart clusters={clusters} />
          </div>
        </div>

        {/* Card 3 — Radial depth */}
        <div className="bezel">
          <div className="bezel-inner" style={{ padding: "20px" }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--muted)", marginBottom: 16, letterSpacing: "0.06em" }}>
              Micro-Cluster Depth
            </div>
            <RadialDepthChart clusters={clusters} />
          </div>
        </div>

      </div>
    </motion.div>
  );
}
