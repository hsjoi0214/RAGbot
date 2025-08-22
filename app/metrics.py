# app/metrics.py
from __future__ import annotations

from typing import cast
from prometheus_client import Counter, Histogram, start_http_server, REGISTRY
from .config import cfg

_started = False

def _start_server_once():
    global _started
    if not _started and cfg.METRICS_ENABLED:
        try:
            start_http_server(cfg.METRICS_PORT)
            _started = True
        except OSError:
            # Already started (common on Streamlit reruns)
            _started = True

# Start the server at import time (safe due to guard)
_start_server_once()

def _get_or_create_counter(name: str, documentation: str, labelnames=()):
    try:
        return Counter(name, documentation, labelnames)
    except ValueError:
        collector = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        # Runtime safety: if it's not a Counter, raise to avoid weirdness
        if not isinstance(collector, Counter):
            raise RuntimeError(f"Collector '{name}' already registered but is not a Counter")
        return cast(Counter, collector)

def _get_or_create_histogram(name: str, documentation: str, labelnames=(), buckets=None):
    try:
        return Histogram(name, documentation, labelnames, buckets=buckets) if buckets else Histogram(name, documentation, labelnames)
    except ValueError:
        collector = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        if not isinstance(collector, Histogram):
            raise RuntimeError(f"Collector '{name}' already registered but is not a Histogram")
        return cast(Histogram, collector)

# --- metrics (typed) ---
REQS = _get_or_create_counter("rag_requests_total", "Total requests", ["route"])
ERRS = _get_or_create_counter("rag_errors_total", "Errors by type", ["route", "type"])

LAT_E2E = _get_or_create_histogram(
    "rag_latency_seconds",
    "End-to-end latency",
    ["route"],
    buckets=[0.2, 0.5, 1, 2, 3, 5, 8, 13],
)

# Optional token histograms; safe even if you don’t observe them yet
TOK_IN  = _get_or_create_histogram("rag_tokens_in",  "Prompt tokens",     ["model"], buckets=[128,256,512,1024,2048,4096])
TOK_OUT = _get_or_create_histogram("rag_tokens_out", "Completion tokens", ["model"], buckets=[64,128,256,512,1024])
