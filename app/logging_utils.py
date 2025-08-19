# app/logging_utils.py
import json, re, time, uuid

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def redact(s: str) -> str:
    if not isinstance(s, str):
        return s
    return _EMAIL.sub("<email>", s)

def log_event(kind: str, **fields):
    evt = {"ts": int(time.time() * 1000), "kind": kind, "event_id": str(uuid.uuid4())}
    for k, v in fields.items():
        if isinstance(v, str):
            evt[k] = redact(v)
        else:
            evt[k] = v
    print(json.dumps(evt, ensure_ascii=False))
