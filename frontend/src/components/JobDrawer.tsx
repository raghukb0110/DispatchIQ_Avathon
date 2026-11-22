import type { AllocationResult, Assignment, Scenario } from "../types";
import { ALGO_META, ALGOS, PRIORITY_COLORS, fmtTime } from "../utils";

interface Props {
  scenario: Scenario;
  jobId: string;
  results: Record<string, AllocationResult>;
  onClose: () => void;
}

export default function JobDrawer({ scenario, jobId, results, onClose }: Props) {
  const job = scenario.jobs.find((j) => j.id === jobId);
  if (!job) return null;
  const picks = ALGOS.map((a) => results[a]?.assignments.find((x) => x.job_id === jobId)?.technician_id);
  const same = picks[0] && picks[0] === picks[1];

  return (
    <aside className="drawer">
      <div className="drawer-head">
        <div>
          <div className="eyebrow">Job inspector</div>
          <h2>{job.id}</h2>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
      </div>
      <div className="chips">
        <span className="chip">{job.required_skill}</span>
        <span className="chip">level ≥ {job.min_skill_level}</span>
        <span className="chip" style={{ borderColor: PRIORITY_COLORS[job.priority], color: PRIORITY_COLORS[job.priority] }}>
          priority {job.priority}
        </span>
        <span className="chip">{job.duration} min</span>
      </div>
      <div className="job-times">
        <div><span className="muted">requested</span><b>{fmtTime(job.requested_at)}</b></div>
        <div><span className="muted">window</span><b>{fmtTime(job.window_start)}–{fmtTime(job.window_end)}</b></div>
      </div>
      <div className={`agree ${same ? "yes" : "no"}`}>
        {same ? `Both algorithms chose ${picks[0]}` : "The algorithms disagree on this job"}
      </div>

      {ALGOS.map((algo) => {
        const r = results[algo];
        const a = r?.assignments.find((x) => x.job_id === jobId);
        const u = r?.unassigned.find((x) => x.job_id === jobId);
        return (
          <div key={algo} className="decision" style={{ ["--accent" as string]: ALGO_META[algo].color }}>
            <div className="decision-head">
              <b>{ALGO_META[algo].label}</b>
              <span className="muted">{ALGO_META[algo].short}</span>
            </div>
            {a ? <AssignmentView a={a} /> : <p className="warn">Unassigned · {u?.reason}</p>}
          </div>
        );
      })}
    </aside>
  );
}

function AssignmentView({ a }: { a: Assignment }) {
  const e = a.explanation;
  const maxAbs = Math.max(...(e.top_factors ?? []).map((f) => Math.abs(f.contribution)), 1e-9);
  return (
    <>
      <div className="assign-line">
        <span className="tech-badge">{a.technician_id}</span>
        <span>arrives {fmtTime(a.arrival)} · starts {fmtTime(a.start)} · {a.travel_km.toFixed(1)} km</span>
      </div>
      <p className="summary">{e.summary}</p>

      {e.top_factors && (
        <>
          <div className="sub">Why the model chose it <span className="muted">(SHAP)</span></div>
          {e.top_factors.map((f) => (
            <div key={f.feature} className="factor">
              <span className="factor-name">{f.feature}<span className="muted"> = {f.value}</span></span>
              <span className="factor-bar">
                <span className={f.contribution >= 0 ? "pos" : "neg"} style={{ width: `${(Math.abs(f.contribution) / maxAbs) * 100}%` }} />
              </span>
              <span className="factor-val">{f.contribution > 0 ? "+" : ""}{f.contribution.toFixed(2)}</span>
            </div>
          ))}
        </>
      )}
      {e.ranking && (
        <>
          <div className="sub">Candidate ranking</div>
          <div className="ranking">
            {e.ranking.map((r, i) => (
              <div key={r.technician_id} className="rank-row">
                <span className="muted">#{i + 1}</span>
                <span>{r.technician_id}</span>
                <span className="conf"><span style={{ width: `${r.confidence * 100}%` }} /></span>
                <span>{(r.confidence * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </>
      )}
      {e.alternatives && e.alternatives.length > 0 && (
        <>
          <div className="sub">Round {e.round} · rank #{e.rank_by_own_cost} by own cost · alternatives</div>
          <div className="chips">
            {e.alternatives.map((alt) => (
              <span key={alt.technician_id} className="chip">{alt.technician_id} · {alt.cost.toFixed(1)}</span>
            ))}
          </div>
        </>
      )}
      <div className="sub">Cost breakdown <span className="muted">(total {a.cost.toFixed(1)})</span></div>
      <div className="chips">
        {Object.entries(a.cost_breakdown).map(([k, v]) => (
          <span key={k} className={`chip ${v < 0 ? "good" : ""}`}>{k} {v.toFixed(1)}</span>
        ))}
      </div>
      {Object.keys(e.rejected).length > 0 && (
        <p className="hint">
          Ruled out by hard constraints: {Object.entries(e.rejected).map(([k, v]) => `${k} ×${v}`).join(", ")}
        </p>
      )}
    </>
  );
}
