import type { Assignment, Job, Technician } from "./types";

export type Algo = "hungarian" | "ml";
export const ALGOS: Algo[] = ["hungarian", "ml"];

export const ALGO_META: Record<Algo, { label: string; short: string; tag: string; color: string }> = {
  hungarian: { label: "Hungarian", short: "Batch", tag: "whole-day plan", color: "#38bdf8" },
  ml: { label: "ML Dispatcher", short: "Online", tag: "per request", color: "#f472b6" },
};

export const DAY_START = 420; // 07:00
export const DAY_END = 1200; // 20:00
export const CITY_SPEED_KMH = 25;

const PALETTE = [
  "#f87171", "#fb923c", "#facc15", "#a3e635", "#34d399", "#22d3ee", "#60a5fa", "#a78bfa",
  "#f472b6", "#e879f9", "#fda4af", "#fdba74", "#86efac", "#67e8f9", "#c4b5fd", "#fcd34d",
];

export function techColor(id: string): string {
  const n = parseInt(id.replace(/\D/g, ""), 10) || 0;
  return PALETTE[n % PALETTE.length];
}

export const PRIORITY_COLORS = ["", "#64748b", "#22d3ee", "#a3e635", "#fb923c", "#ef4444"];

export function fmtTime(minutes: number): string {
  const m = Math.round(minutes);
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

export const pct = (v: number, digits = 0) => `${v.toFixed(digits)}%`;

export interface MetricInfo {
  label: string;
  unit?: string;
  better: "high" | "low";
  hint: string;
}

export const METRICS: Record<string, MetricInfo> = {
  fulfillment_rate: { label: "Jobs fulfilled", unit: "%", better: "high", hint: "Share of all jobs that got a technician" },
  priority_weighted_fulfillment: { label: "Priority-weighted", unit: "%", better: "high", hint: "Fulfilment where a P5 job counts 5x a P1 job" },
  high_priority_fulfillment: { label: "High-priority served", unit: "%", better: "high", hint: "Share of P4-P5 jobs served" },
  avg_travel_km: { label: "Travel per job", unit: "km", better: "low", hint: "Average drive to each assigned job" },
  avg_idle_min: { label: "Idle before job", unit: "min", better: "low", hint: "Unproductive waiting before each job" },
  utilization: { label: "Utilisation", unit: "%", better: "high", hint: "Share of shift time spent driving or working" },
  workload_std: { label: "Workload spread", unit: "σ", better: "low", hint: "Std-dev of jobs per technician (lower = fairer)" },
  techs_used: { label: "Technicians used", better: "high", hint: "Technicians with at least one job" },
  runtime_ms: { label: "Compute time", unit: "ms", better: "low", hint: "Wall-clock time for the whole day" },
};

export type JobStatus = "hidden" | "waiting" | "planned" | "active" | "done" | "unassigned";

/** Status of a job at playback time `t` for one algorithm. */
export function jobStatus(job: Job, a: Assignment | undefined, t: number): JobStatus {
  if (job.requested_at > t) return "hidden";
  if (!a) return job.window_end < t ? "unassigned" : "waiting";
  if (a.end <= t) return "done";
  if (a.start <= t) return "active";
  return "planned";
}

/** Technician position at time `t`, interpolated along the leg being driven. */
export function techPosition(
  tech: Technician,
  route: Assignment[],
  jobs: Map<string, Job>,
  t: number,
): [number, number] {
  let prev: [number, number] = [tech.lat, tech.lng];
  for (const a of route) {
    const j = jobs.get(a.job_id)!;
    const travel = (a.travel_km / CITY_SPEED_KMH) * 60;
    const depart = a.arrival - travel;
    if (t < depart) break;
    if (t < a.arrival && travel > 0) {
      const f = (t - depart) / travel;
      return [prev[0] + (j.lat - prev[0]) * f, prev[1] + (j.lng - prev[1]) * f];
    }
    prev = [j.lat, j.lng];
  }
  return prev;
}

export function routesByTech(assignments: Assignment[]): Map<string, Assignment[]> {
  const m = new Map<string, Assignment[]>();
  for (const a of assignments) {
    if (!m.has(a.technician_id)) m.set(a.technician_id, []);
    m.get(a.technician_id)!.push(a);
  }
  m.forEach((r) => r.sort((x, y) => x.start - y.start));
  return m;
}
