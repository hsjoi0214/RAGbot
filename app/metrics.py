# app/metrics.py
"""Prometheus metrics setup and helpers.

This module defines reusable Prometheus metrics (counters and histograms) and
starts a single HTTP exporter for scraping. It is safe under hot-reload
environments (e.g., Streamlit) via an idempotent server starter.

Notes:
    - Import-time side effect: the exporter is started once if metrics are enabled.
    - Idempotency: helpers return an existing collector if the name is already
      registered; if a different type is found under the same name, we raise.
    - Prometheus client registry is process-global; in multi-process deployments
      use a Pushgateway or ensure only one process exposes the HTTP endpoint.
"""

from __future__ import annotations

from typing import cast
from prometheus_client import Counter, Histogram, start_http_server, REGISTRY
from .config import cfg

_started = False

def _start_server_once():
    """Start the Prometheus exporter exactly once if enabled.

    Behavior:
        - If metrics are disabled, does nothing.
        - If the port is already bound (common on hot reloads), treat that as
          "already started" and keep going.
    Side effects:
        - Binds an HTTP server on ``cfg.METRICS_PORT`` for Prometheus to scrape.
        - Sets the module-global ``_started`` flag.
    """
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
    """Create a Counter or return the already-registered one with the same name.

    This uses the process-wide default registry. If a collector with the same
    name exists but is *not* a Counter, we raise to avoid silent type drift.

    Args:
        name: Prometheus metric name (snake_case, globally unique in registry).
        documentation: Human-readable description for the metric.
        labelnames: Iterable of label names to define on the counter.

    Returns:
        prometheus_client.Counter: The new or existing Counter.

    Raises:
        RuntimeError: If a collector with ``name`` exists and is not a Counter.
    """
    try:
        return Counter(name, documentation, labelnames)
    except ValueError:
        # Accessing a private map is pragmatic here for idempotency under reloads.
        collector = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        # Runtime safety: if it's not a Counter, raise to avoid weirdness
        if not isinstance(collector, Counter):
            raise RuntimeError(f"Collector '{name}' already registered but is not a Counter")
        return cast(Counter, collector)

def _get_or_create_histogram(name: str, documentation: str, labelnames=(), buckets=None):
    """Create a Histogram or return the already-registered one with the same name.

    Args:
        name: Prometheus metric name (snake_case, globally unique in registry).
        documentation: Human-readable description for the metric.
        labelnames: Iterable of label names to define on the histogram.
        buckets: Optional list of bucket boundaries (ascending). If None, uses
            the Prometheus client's defaults.

    Returns:
        prometheus_client.Histogram: The new or existing Histogram.

    Raises:
        RuntimeError: If a collector with ``name`` exists and is not a Histogram.
    """
    try:
        return Histogram(name, documentation, labelnames, buckets=buckets) if buckets else Histogram(name, documentation, labelnames)
    except ValueError:
        # Accessing a private map is pragmatic here for idempotency under reloads.
        collector = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        if not isinstance(collector, Histogram):
            raise RuntimeError(f"Collector '{name}' already registered but is not a Histogram")
        return cast(Histogram, collector)

# --- metrics (typed) ---
# Request and error counters (low cardinality labels recommended).
REQS = _get_or_create_counter("rag_requests_total", "Total requests", ["route"])
ERRS = _get_or_create_counter("rag_errors_total", "Errors by type", ["route", "type"])

# End-to-end latency buckets tuned for interactive workloads; adjust from real data.
LAT_E2E = _get_or_create_histogram(
    "rag_latency_seconds",
    "End-to-end latency",
    ["route"],
    buckets=[0.2, 0.5, 1, 2, 3, 5, 8, 13],
)

# Optional token histograms; safe even if you don’t observe them yet.
# Label by model so dashboards can slice usage per provider/model.
TOK_IN  = _get_or_create_histogram("rag_tokens_in",  "Prompt tokens",     ["model"], buckets=[128,256,512,1024,2048,4096])
TOK_OUT = _get_or_create_histogram("rag_tokens_out", "Completion tokens", ["model"], buckets=[64,128,256,512,1024])
