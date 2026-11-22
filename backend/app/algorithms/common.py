import time

from app.core.constraints import PairEval
from app.core.explain import unassigned_reason
from app.core.metrics import compute_metrics
from app.core.state import TechState
from app.models import AllocationResult, Assignment, Job, Scenario, UnassignedJob


def make_assignment(
    state: TechState, job: Job, ev: PairEval, cost: float, breakdown: dict[str, float],
    explanation: dict, score: float | None = None,
) -> Assignment:
    return Assignment(
        technician_id=state.tech.id, job_id=job.id,
        arrival=round(ev.arrival, 2), start=round(ev.start, 2), end=round(ev.end, 2),
        travel_km=round(ev.travel_km, 3), wait_min=round(ev.idle_min, 2),
        cost=cost, cost_breakdown=breakdown, score=score, explanation=explanation,
    )


def finalize(
    name: str, scenario: Scenario, assignments: list[Assignment],
    states: dict[str, TechState], t0: float,
) -> AllocationResult:
    runtime_ms = (time.perf_counter() - t0) * 1000
    done = {a.job_id for a in assignments}
    unassigned = [
        UnassignedJob(job_id=j.id, reason=unassigned_reason(j, scenario.technicians))
        for j in scenario.jobs if j.id not in done
    ]
    return AllocationResult(
        algorithm=name, assignments=assignments, unassigned=unassigned,
        metrics=compute_metrics(scenario, assignments, states, runtime_ms), runtime_ms=runtime_ms,
    )
