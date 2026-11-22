import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip } from "recharts";
import type { CompareResponse } from "../types";
import { ALGO_META, METRICS } from "../utils";

const RADAR = ["fulfillment_rate", "priority_weighted_fulfillment", "high_priority_fulfillment", "utilization", "avg_travel_km", "avg_idle_min", "workload_std"];
const TABLE = [...RADAR.slice(0, 3), "avg_travel_km", "avg_idle_min", "utilization", "workload_std", "techs_used", "runtime_ms"];

function score(metric: string, v: number, other: number) {
  const better = METRICS[metric].better;
  if (better === "high") return Math.max(v, other) > 0 ? (v / Math.max(v, other)) * 100 : 100;
  return v <= 0 ? 100 : (Math.min(v, other) / v) * 100;
}

export default function Scoreboard({ data }: { data: CompareResponse }) {
  const h = data.results.hungarian.metrics;
  const m = data.results.ml.metrics;
  const s = data.summary;

  const radar = RADAR.map((k) => ({
    metric: METRICS[k].label,
    hungarian: Math.round(score(k, h[k], m[k])),
    ml: Math.round(score(k, m[k], h[k])),
  }));

  const wins = { hungarian: 0, ml: 0 };
  TABLE.forEach((k) => {
    if (h[k] === m[k]) return;
    const hBetter = METRICS[k].better === "high" ? h[k] > m[k] : h[k] < m[k];
    wins[hBetter ? "hungarian" : "ml"]++;
  });

  const verdict = [
    s.fulfillment_gap >= 0
      ? `The ML dispatcher served ${s.fulfillment_gap.toFixed(1)} pts more jobs.`
      : `Hungarian served ${(-s.fulfillment_gap).toFixed(1)} pts more jobs.`,
    h.high_priority_fulfillment >= m.high_priority_fulfillment
      ? `Hungarian protected urgent work better (${h.high_priority_fulfillment.toFixed(0)}% vs ${m.high_priority_fulfillment.toFixed(0)}% of P4–P5 jobs) because it sees every request up front.`
      : `ML served more urgent work (${m.high_priority_fulfillment.toFixed(0)}% vs ${h.high_priority_fulfillment.toFixed(0)}% of P4–P5 jobs).`,
    `Both picked the same technician for ${(s.agreement * 100).toFixed(0)}% of jobs - different plans can reach similar outcomes.`,
  ];

  return (
    <div className="scoreboard">
      <div className="score-hero">
        {(["hungarian", "ml"] as const).map((a) => (
          <div key={a} className="score-side" style={{ ["--accent" as string]: ALGO_META[a].color }}>
            <div className="score-name">{ALGO_META[a].label}</div>
            <div className="score-num">{wins[a]}</div>
            <div className="muted">metrics won</div>
          </div>
        ))}
        <div className="score-verdict">
          <div className="eyebrow">Verdict</div>
          <ul>
            {verdict.map((v) => <li key={v}>{v}</li>)}
          </ul>
        </div>
      </div>

      <div className="score-grid">
        <div className="panel">
          <div className="panel-title">Shape of each plan <span className="muted">(100 = best of the two)</span></div>
          <ResponsiveContainer width="100%" height={340}>
            <RadarChart data={radar} outerRadius="72%">
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#cbd5e1", fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
              <Radar name={ALGO_META.hungarian.label} dataKey="hungarian" stroke={ALGO_META.hungarian.color} fill={ALGO_META.hungarian.color} fillOpacity={0.25} />
              <Radar name={ALGO_META.ml.label} dataKey="ml" stroke={ALGO_META.ml.color} fill={ALGO_META.ml.color} fillOpacity={0.25} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <div className="panel-title">Head to head</div>
          <div className="h2h">
            {TABLE.map((k) => {
              const info = METRICS[k];
              const max = Math.max(h[k], m[k]) || 1;
              const hBetter = info.better === "high" ? h[k] > m[k] : h[k] < m[k];
              const tie = h[k] === m[k];
              const fmt = (v: number) => (k === "techs_used" ? v.toFixed(0) : v.toFixed(k === "runtime_ms" ? 0 : 1));
              return (
                <div className="h2h-row" key={k} title={info.hint}>
                  <span className={`h2h-val left ${!tie && hBetter ? "win" : ""}`}>{fmt(h[k])}</span>
                  <div className="h2h-bars">
                    <div className="bar left" style={{ width: `${(h[k] / max) * 50}%`, background: ALGO_META.hungarian.color }} />
                    <div className="bar right" style={{ width: `${(m[k] / max) * 50}%`, background: ALGO_META.ml.color }} />
                    <span className="h2h-label">
                      {info.label} {info.unit && <span className="muted">{info.unit}</span>} {info.better === "high" ? "↑" : "↓"}
                    </span>
                  </div>
                  <span className={`h2h-val ${!tie && !hBetter ? "win" : ""}`}>{fmt(m[k])}</span>
                </div>
              );
            })}
          </div>
          <p className="hint">
            Compute time is for the whole day. The ML dispatcher spends ≈{(m.runtime_ms / Math.max(m.jobs_total, 1)).toFixed(1)} ms per request and
            needs no batch; Hungarian needs every request before it can plan.
          </p>
        </div>
      </div>
    </div>
  );
}
