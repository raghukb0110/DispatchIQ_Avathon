import numpy as np

from app.core.state import TechState
from app.models import Assignment, Scenario


def compute_metrics(
    scenario: Scenario, assignments: list[Assignment], states: dict[str, TechState], runtime_ms: float
) -> dict[str, float]:
    jobs = {j.id: j for j in scenario.jobs}
    n_jobs = len(jobs)
    n_assigned = len(assignments)
    prio_total = sum(j.priority for j in jobs.values()) or 1
    prio_done = sum(jobs[a.job_id].priority for a in assignments)
    high = [j for j in jobs.values() if j.priority >= 4]
    high_done = sum(1 for a in assignments if jobs[a.job_id].priority >= 4)
    shift_total = sum(t.shift_end - t.shift_start for t in scenario.technicians) or 1
    per_tech = [s.jobs_done for s in states.values()]

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "jobs_total": n_jobs,
        "jobs_assigned": n_assigned,
        "fulfillment_rate": 100.0 * n_assigned / n_jobs if n_jobs else 0.0,
        "priority_weighted_fulfillment": 100.0 * prio_done / prio_total,
        "high_priority_fulfillment": 100.0 * high_done / len(high) if high else 100.0,
        "total_travel_km": sum(a.travel_km for a in assignments),
        "avg_travel_km": mean([a.travel_km for a in assignments]),
        "avg_idle_min": mean([a.wait_min for a in assignments]),
        "avg_start_delay_min": mean([a.start - jobs[a.job_id].window_start for a in assignments]),
        "utilization": 100.0 * sum(s.busy_min for s in states.values()) / shift_total,
        "workload_std": float(np.std(per_tech)) if per_tech else 0.0,
        "techs_used": sum(1 for n in per_tech if n > 0),
        "total_cost": sum(a.cost for a in assignments),
        "runtime_ms": runtime_ms,
    }
