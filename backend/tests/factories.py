from app.models import Job, Technician


def tech(tid="T1", lat=12.97, lng=77.59, skills=("electrical",), level=2, start=480, end=1020, max_jobs=5):
    return Technician(id=tid, name=tid, lat=lat, lng=lng, skills=list(skills), skill_level=level,
                      shift_start=start, shift_end=end, max_jobs=max_jobs)


def job(jid="J1", lat=12.98, lng=77.60, skill="electrical", level=1, requested=480,
        ws=540, we=660, duration=60, priority=3):
    return Job(id=jid, lat=lat, lng=lng, required_skill=skill, min_skill_level=level, requested_at=requested,
               window_start=ws, window_end=we, duration=duration, priority=priority)
