# app/logging_utils.py
"""Lightweight structured logging utilities.

This module provides helpers for emitting JSON-formatted log events with
automatic field redaction. It is designed for simplicity and minimal runtime
dependencies.

Notes:
    - Each event is printed to stdout as a single JSON object.
    - A millisecond timestamp and a unique event ID are always included.
    - Email addresses in string fields are redacted automatically.
"""

import json, re, time, uuid

# Precompiled regex for email address redaction.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def redact(s: str) -> str:
    """Redact sensitive substrings (currently only email addresses).

    Args:
        s: Input string to sanitize.

    Returns:
        The same string with email addresses replaced by ``"<email>"``.
        Non-string inputs are returned unchanged.
    """
    if not isinstance(s, str):
        return s
    return _EMAIL.sub("<email>", s)


def log_event(kind: str, **fields):
    """Emit a structured log event.

    Constructs a JSON object with a timestamp, event type, and unique event ID,
    enriched with user-provided fields. String fields are sanitized via
    ``redact()``.

    Args:
        kind: Logical type of the event (e.g., "request", "error", "metric").
        **fields: Arbitrary event fields (values can be any JSON-serializable type).

    Side effects:
        Prints the JSON log line to stdout.
    """
    evt = {"ts": int(time.time() * 1000), "kind": kind, "event_id": str(uuid.uuid4())}
    for k, v in fields.items():
        if isinstance(v, str):
            evt[k] = redact(v)
        else:
            evt[k] = v
    print(json.dumps(evt, ensure_ascii=False))
