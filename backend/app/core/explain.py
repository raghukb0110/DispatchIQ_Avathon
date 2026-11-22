from collections import Counter

from app.core.constraints import PairEval, is_qualified
from app.models import Job, Technician


def rejection_summary(rejections: list[PairEval]) -> dict[str, int]:
    return dict(Counter(r.code for r in rejections))


def unassigned_reason(job: Job, techs: list[Technician]) -> str:
    if not any(job.required_skill in t.skills for t in techs):
        return f"No technician has skill '{job.required_skill}'"
    if not any(is_qualified(t, job) for t in techs):
        return f"No technician has '{job.required_skill}' at level >= {job.min_skill_level}"
    return "All qualified technicians were committed to other jobs or could not reach it within its time window"
