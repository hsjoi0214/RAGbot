# app/feedback.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data/feedback.sqlite")

def _conn():
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
    with _conn() as con:
        con.execute("INSERT INTO feedback(ts, request_id, rating, comment) VALUES(?,?,?,?)",
                    (ts, request_id, rating, comment or ""))
