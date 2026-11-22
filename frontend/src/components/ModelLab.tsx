import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ModelCard } from "../types";
import { ALGO_META, ALGOS, METRICS } from "../utils";

const PIPELINE = [
  { icon: "🎲", title: "Simulate", text: "200 seeded synthetic days" },
  { icon: "🧮", title: "Teacher", text: "Hungarian solves each day" },
  { icon: "🔁", title: "Replay", text: "Label the teacher's pick per request" },
  { icon: "🌲", title: "Train", text: "LambdaRank, grouped by decision" },
  { icon: "🧪", title: "Validate", text: "Split by day, never by row" },
  { icon: "🚚", title: "Serve", text: "Rank feasible techs online" },
];

export default function ModelLab({ card }: { card: ModelCard | null }) {
  if (!card) return <div className="panel">Loading model card…</div>;

  const importance = Object.entries(card.feature_importance as Record<string, number>)
    .slice(0, 10)
    .map(([feature, v]) => ({ feature, value: Number((v * 100).toFixed(1)) }));
  const business = card.business_eval_holdout as Record<string, Record<string, number>>;
  const lift = card.test.top1_accuracy / card.test.cost_baseline_top1;

  return (
    <div className="lab">
      <div className="pipeline">
        {PIPELINE.map((s, i) => (
          <div key={s.title} className="pipe-step">
            <div className="pipe-icon">{s.icon}</div>
            <b>{s.title}</b>
            <span className="muted">{s.text}</span>
            {i < PIPELINE.length - 1 && <span className="pipe-arrow">→</span>}
          </div>
        ))}
      </div>

      <div className="kpis">
        <div className="kpi"><div className="kpi-label">Served model</div><div className="kpi-value small">{card.model_kind}</div></div>
        <div className="kpi">
          <div className="kpi-label">Matches the optimiser (top-1)</div>
          <div className="kpi-value">{(card.test.top1_accuracy * 100).toFixed(1)}%</div>
          <div className="muted">vs {(card.test.cost_baseline_top1 * 100).toFixed(1)}% for “pick the cheapest” · {lift.toFixed(2)}× lift</div>
        </div>
        <div className="kpi"><div className="kpi-label">Test AUC</div><div className="kpi-value">{card.test.auc.toFixed(3)}</div></div>
        <div className="kpi">
          <div className="kpi-label">Training data</div>
          <div className="kpi-value small">{card.data.decisions.toLocaleString()} decisions</div>
          <div className="muted">{card.data.rows.toLocaleString()} rows · {card.data.train_scenarios}/{card.data.test_scenarios} days train/test</div>
        </div>
      </div>

      <div className="score-grid">
        <div className="panel">
          <div className="panel-title">What the model looks at <span className="muted">(gain %)</span></div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={importance} layout="vertical" margin={{ left: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" tick={{ fill: "#94a3b8" }} />
              <YAxis type="category" dataKey="feature" width={140} tick={{ fill: "#cbd5e1", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
              <Bar dataKey="value" fill={ALGO_META.ml.color} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <div className="panel-title">Model selection <span className="muted">(3-fold CV, grouped by day)</span></div>
          <table className="table">
            <thead><tr><th>Model</th><th>AUC</th><th>Top-1</th></tr></thead>
            <tbody>
              {Object.entries(card.cv as Record<string, Record<string, number>>).map(([k, v]) => (
                <tr key={k} className={k === card.model_kind ? "chosen" : ""}>
                  <td>{k}{k === card.model_kind && " ✓"}</td>
                  <td>{v.auc.toFixed(3)}</td>
                  <td>{(v.top1_accuracy * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">Selected on top-1, not AUC: the dispatcher only compares candidates within one request.</p>

          <div className="panel-title" style={{ marginTop: 16 }}>Held-out days: outcomes</div>
          <table className="table">
            <thead>
              <tr><th>Metric</th>{ALGOS.map((a) => <th key={a} style={{ color: ALGO_META[a].color }}>{ALGO_META[a].label}</th>)}</tr>
            </thead>
            <tbody>
              {Object.keys(business.hungarian).map((k) => (
                <tr key={k}>
                  <td>{METRICS[k]?.label ?? k}</td>
                  {ALGOS.map((a) => <td key={a}>{business[a][k].toFixed(1)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel note">
        <b>Intended use.</b> {card.intended_use}
        <br />
        <b>Limitations.</b> {card.limitations}
      </div>
    </div>
  );
}
