# app/feedback.py
"""User feedback persistence.

This module stores feedback events (rating + optional comment) into a local
SQLite database. The schema is created automatically if it does not exist.

Notes:
    - Database file lives at ``data/feedback.sqlite`` by default.
    - Simple schema: auto-increment id, timestamp, request ID, rating, comment.
    - Ratings are expected to be ``'up'`` or ``'down'``, but not enforced here.
    - Writes are auto-committed using context managers.
"""

import sqlite3
from pathlib import Path

# Path to the SQLite database file.
DB_PATH = Path("data/feedback.sqlite")


def _conn():
    """Create and return a SQLite connection with feedback schema ensured.

    Side effects:
        - Ensures the parent directory for the DB exists.
        - Creates the ``feedback`` table if it does not exist.

    Returns:
        sqlite3.Connection: Open connection to the feedback database.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER,
        request_id TEXT,
        rating TEXT,      -- 'up' | 'down'
        comment TEXT
    )""")
    return con


def record_feedback(ts: int, request_id: str, rating: str, comment: str | None):
    """Insert a feedback record into the database.

    Args:
        ts: Event timestamp (e.g., epoch milliseconds).
        request_id: Identifier of the request/session being rated.
        rating: Feedback rating, conventionally 'up' or 'down'.
        comment: Optional free-text comment. If None, stored as empty string.

    Side effects:
        Persists the feedback row into the ``feedback`` table.
    """
    with _conn() as con:
        con.execute("INSERT INTO feedback(ts, request_id, rating, comment) VALUES(?,?,?,?)",
                    (ts, request_id, rating, comment or ""))
