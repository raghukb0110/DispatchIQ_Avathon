"""Independent re-check of hard constraints by replaying each technician's route."""
from app.core.constraints import is_qualified
from app.core.geo import haversine_km, travel_minutes
from app.models import Assignment, Scenario

EPS = 0.02  # assignment times are rounded to 2 decimals


def validate_assignments(scenario: Scenario, assignments: list[Assignment]) -> list[str]:
    errors: list[str] = []
    jobs = {j.id: j for j in scenario.jobs}
    techs = {t.id: t for t in scenario.technicians}

    seen: set[str] = set()
    for a in assignments:
        if a.job_id in seen:
            errors.append(f"job {a.job_id} assigned more than once")
        seen.add(a.job_id)

    by_tech: dict[str, list[Assignment]] = {}
    for a in assignments:
        by_tech.setdefault(a.technician_id, []).append(a)

    for tid, route in by_tech.items():
        tech = techs[tid]
        if len(route) > tech.max_jobs:
            errors.append(f"{tid} exceeds max jobs")
        lat, lng, free_at = tech.lat, tech.lng, float(tech.shift_start)
        for a in sorted(route, key=lambda x: x.start):
            job = jobs[a.job_id]
            if not is_qualified(tech, job):
                errors.append(f"{tid} not qualified for {job.id}")
            earliest = max(free_at, job.requested_at) + travel_minutes(haversine_km(lat, lng, job.lat, job.lng))
            if a.start + EPS < max(earliest, job.window_start):
                errors.append(f"{tid} cannot start {job.id} at {a.start:.1f}")
            if a.start > job.window_end + EPS:
                errors.append(f"{job.id} starts after its window")
            if abs(a.end - (a.start + job.duration)) > EPS:
                errors.append(f"{job.id} has wrong end time")
            if a.end > tech.shift_end + EPS:
                errors.append(f"{job.id} ends after {tid}'s shift")
            lat, lng, free_at = job.lat, job.lng, a.end
    return errors
