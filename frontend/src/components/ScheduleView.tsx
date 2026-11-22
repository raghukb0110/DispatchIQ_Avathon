import type { AllocationResult, Scenario } from "../types";
import { ALGO_META, ALGOS, DAY_END, DAY_START, CITY_SPEED_KMH, PRIORITY_COLORS, fmtTime, routesByTech, techColor } from "../utils";

interface Props {
  scenario: Scenario;
  results: Record<string, AllocationResult>;
  time: number;
  selectedJob: string | null;
  onSelectJob: (id: string) => void;
}

const span = DAY_END - DAY_START;
const x = (m: number) => `${((Math.min(Math.max(m, DAY_START), DAY_END) - DAY_START) / span) * 100}%`;
const w = (a: number, b: number) => `${(Math.max(0, Math.min(b, DAY_END) - Math.max(a, DAY_START)) / span) * 100}%`;
const HOURS = Array.from({ length: span / 60 + 1 }, (_, i) => DAY_START + i * 60);

export default function ScheduleView({ scenario, results, time, selectedJob, onSelectJob }: Props) {
  const routes = Object.fromEntries(ALGOS.map((a) => [a, routesByTech(results[a]?.assignments ?? [])]));
  const priority = new Map(scenario.jobs.map((j) => [j.id, j.priority]));

  return (
    <div className="schedule">
      <div className="schedule-legend">
        <span><i className="lg travel" /> driving</span>
        <span><i className="lg idle" /> idle / waiting</span>
        <span><i className="lg job" /> on the job (colour = priority P1→P5)</span>
        <span><i className="lg shift" /> shift</span>
        {ALGOS.map((a) => (
          <span key={a}><i className="lg lane" style={{ background: ALGO_META[a].color }} /> {ALGO_META[a].label} lane</span>
        ))}
      </div>
      <div className="gantt">
        <div className="gantt-axis">
          <div className="gantt-label" />
          <div className="gantt-track">
            {HOURS.map((h) => (
              <span key={h} style={{ left: x(h) }}>{fmtTime(h)}</span>
            ))}
          </div>
        </div>
        {scenario.technicians.map((t) => (
          <div className="gantt-row" key={t.id}>
            <div className="gantt-label">
              <span className="dot" style={{ background: techColor(t.id) }} />
              {t.id}
              <span className="muted"> L{t.skill_level}</span>
              <div className="gantt-skills">{t.skills.join(" · ")}</div>
            </div>
            <div className="gantt-lanes">
              {ALGOS.map((algo) => {
                const route = routes[algo].get(t.id) ?? [];
                return (
                  <div className="gantt-track lane" key={algo} style={{ ["--accent" as string]: ALGO_META[algo].color }}>
                    <div className="shift" style={{ left: x(t.shift_start), width: w(t.shift_start, t.shift_end) }} />
                    {route.map((a) => {
                      const depart = a.arrival - (a.travel_km / CITY_SPEED_KMH) * 60;
                      return (
                        <div key={a.job_id}>
                          <div className="seg travel" style={{ left: x(depart), width: w(depart, a.arrival) }} />
                          <div className="seg idle" style={{ left: x(a.arrival), width: w(a.arrival, a.start) }} />
                          <button
                            className={`seg job ${selectedJob === a.job_id ? "sel" : ""}`}
                            style={{ left: x(a.start), width: w(a.start, a.end), background: PRIORITY_COLORS[priority.get(a.job_id) ?? 1] }}
                            title={`${a.job_id} · ${fmtTime(a.start)}–${fmtTime(a.end)} · ${a.travel_km.toFixed(1)} km`}
                            onClick={() => onSelectJob(a.job_id)}
                          />
                        </div>
                      );
                    })}
                    <span className="lane-count">{route.length}</span>
                  </div>
                );
              })}
              <div className="now" style={{ left: x(time) }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
