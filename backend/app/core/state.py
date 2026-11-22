from dataclasses import dataclass, field

from app.models import Job, Technician


@dataclass
class TechState:
    """Mutable per-technician state while an allocator builds routes (append-only)."""

    tech: Technician
    lat: float
    lng: float
    available_at: float
    jobs_done: int = 0
    busy_min: float = 0.0
    route: list[str] = field(default_factory=list)


def init_states(techs: list[Technician]) -> dict[str, TechState]:
    return {t.id: TechState(tech=t, lat=t.lat, lng=t.lng, available_at=t.shift_start) for t in techs}


def commit(state: TechState, job: Job, end: float, travel_min: float) -> None:
    state.lat, state.lng = job.lat, job.lng
    state.available_at = end
    state.jobs_done += 1
    state.busy_min += travel_min + job.duration
    state.route.append(job.id)
