import json
import numpy as np
from pathlib import Path
from collections import Counter

raw_spans = [json.loads(l) for l in Path("data/raw_spans.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
spans = [json.loads(l) for l in Path("data/spans.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
scores = json.loads(Path("data/triage_scores.json").read_text(encoding="utf-8"))
prob_spans = [json.loads(l) for l in Path("data/problematic_spans.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
macro = json.loads(Path("data/macro_clusters.json").read_text(encoding="utf-8"))
micro = json.loads(Path("data/micro_clusters.json").read_text(encoding="utf-8"))

print(f"Total raw: {len(raw_spans)}, spans: {len(spans)}, prob_spans: {len(prob_spans)}")

# 1. Truncated check
raw_by_id = {r.get("span_id"): r for r in raw_spans}
span_by_id = {s.get("span_id"): s for s in spans}

trunc_in_raw = [r for r in raw_spans if "length" in (r.get("response_finish_reasons") or [])]
print(f"\n--- TRUNCATED OUTPUT EVALUATION ---")
print(f"Total spans with finish_reason == 'length' in raw_spans: {len(trunc_in_raw)}")
for i, r in enumerate(trunc_in_raw[:6]):
    out_msgs = r.get("output_messages") or (r.get("span_attributes") or {}).get("gen_ai.output.messages") or ""
    tokens = r.get("usage_output_tokens") or (r.get("span_attributes") or {}).get("gen_ai.usage.output_tokens")
    print(f"  [{i+1}] span_id: {r.get('span_id')} | tokens: {tokens} | finish_reasons: {r.get('response_finish_reasons')} | snippet: {str(out_msgs)[-100:]!r}")

# 2. Latency check
print(f"\n--- LATENCY EVALUATION ---")
valid_durs = [s.get("duration_ms") or 0.0 for s in spans if (s.get("duration_ms") or 0) > 0]
p95 = float(np.percentile(valid_durs, 95))
mean_dur = float(np.mean(valid_durs))
median_dur = float(np.median(valid_durs))
p75 = float(np.percentile(valid_durs, 75))
p99 = float(np.percentile(valid_durs, 99))
print(f"Global Duration stats (ms): mean={mean_dur:.1f}, median={median_dur:.1f}, P75={p75:.1f}, P95={p95:.1f}, P99={p99:.1f}")

op_stats = {}
for s in spans:
    op = s.get("operation_name", "unknown")
    dur = s.get("duration_ms") or 0.0
    op_stats.setdefault(op, []).append(dur)

for op, durs in sorted(op_stats.items()):
    print(f"  Op '{op}' (n={len(durs)}): mean={np.mean(durs):.1f}ms, median={np.median(durs):.1f}ms, P95={np.percentile(durs, 95):.1f}ms, max={max(durs):.1f}ms")

high_lat = [s for s in spans if (s.get("duration_ms") or 0) > p95]
print(f"\nSpans flagged as high latency (> P95={p95:.1f}ms): {len(high_lat)}")
for s in high_lat:
    print(f"  id: {s['span_id'][:12]} | op: {s['operation_name']} | dur: {s['duration_ms']:.1f}ms | status: {s['status']} | model: {s['model']}")

# 3. Macro clusters breakdown
print(f"\n--- MACRO CLUSTERS BREAKDOWN ---")
for cid, c in sorted(macro.items(), key=lambda x: -x[1]["size"]):
    sids = c["span_ids"]
    members = [span_by_id[sid] for sid in sids if sid in span_by_id]
    ops = Counter(m.get("operation_name") for m in members)
    statuses = Counter(m.get("status") for m in members)
    prob_types = Counter()
    for m in members:
        for pt in m.get("problem_types", []):
            prob_types[pt] += 1
    excs = Counter(m.get("exception_type") for m in members if m.get("exception_type"))
    models = Counter(m.get("model") for m in members if m.get("model"))
    print(f"\nMacro Cluster [{cid}] label: '{c.get('label')}' (size: {c['size']}, noise: {c.get('is_noise')})")
    print(f"  Description: {c.get('description')}")
    print(f"  Ops: {dict(ops)}")
    print(f"  Statuses: {dict(statuses)}")
    print(f"  Problem Types: {dict(prob_types)}")
    print(f"  Exceptions: {dict(excs)}")
    print(f"  Models: {dict(models)}")

# 4. Micro clusters breakdown
print(f"\n--- MICRO CLUSTERS BREAKDOWN ---")
for mid, subdict in micro.items():
    if not subdict:
        continue
    print(f"\nMicro for Macro [{mid}] ('{macro.get(mid, {}).get('label')}'):")
    for subid, subc in subdict.items():
        sids = subc["span_ids"]
        members = [span_by_id[sid] for sid in sids if sid in span_by_id]
        print(f"  Subcluster [{mid}.{subid}] label: '{subc.get('label')}' (size: {subc['size']}, noise: {subc.get('is_noise')})")
        print(f"    Desc: {subc.get('description')}")
        excs = Counter(m.get("exception_type") for m in members if m.get("exception_type"))
        print(f"    Exceptions: {dict(excs)}")
        inputs_sample = [m.get("input_summary", "")[:50] for m in members[:2]]
        print(f"    Inputs sample: {inputs_sample}")
