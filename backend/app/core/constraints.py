"""Hard constraints. A pair is either feasible (with its timing) or rejected with a reason."""
from dataclasses import dataclass

from app.core.geo import fmt_time, haversine_km, travel_minutes
from app.core.state import TechState
from app.models import Job, Technician


@dataclass
class PairEval:
    feasible: bool
    code: str | None = None  # skill | level | capacity | late | shift
    reason: str | None = None
    travel_km: float = 0.0
    travel_min: float = 0.0
    arrival: float = 0.0
    start: float = 0.0
    end: float = 0.0
    idle_min: float = 0.0


def is_qualified(tech: Technician, job: Job) -> bool:
    return job.required_skill in tech.skills and tech.skill_level >= job.min_skill_level


def evaluate_pair(state: TechState, job: Job) -> PairEval:
    tech = state.tech
    if job.required_skill not in tech.skills:
        return PairEval(False, "skill", f"{tech.id}: missing skill '{job.required_skill}'")
    if tech.skill_level < job.min_skill_level:
        return PairEval(False, "level", f"{tech.id}: skill level {tech.skill_level} < required {job.min_skill_level}")
    if state.jobs_done >= tech.max_jobs:
        return PairEval(False, "capacity", f"{tech.id}: reached max jobs ({tech.max_jobs})")

    km = haversine_km(state.lat, state.lng, job.lat, job.lng)
    t_min = travel_minutes(km)
    # A technician cannot head to a job before it has been requested.
    depart = max(state.available_at, job.requested_at)
    arrival = depart + t_min
    start = max(arrival, job.window_start)
    if start > job.window_end:
        return PairEval(
            False, "late",
            f"{tech.id}: earliest arrival {fmt_time(arrival)} is after window closes {fmt_time(job.window_end)}",
        )
    end = start + job.duration
    if end > tech.shift_end:
        return PairEval(
            False, "shift", f"{tech.id}: job would end {fmt_time(end)}, after shift end {fmt_time(tech.shift_end)}"
        )
    idle = start - state.available_at - t_min
    return PairEval(True, None, None, km, t_min, arrival, start, end, max(idle, 0.0))
