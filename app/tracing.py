# app/tracing.py
"""OpenTelemetry tracing setup.

This module configures an OpenTelemetry tracer for the application. It supports
both real OTLP exporters and a no-op fallback for cases where tracing is disabled
or the configured endpoint is unreachable.

Notes:
    - Import-time: no side effects. Tracer is initialized lazily.
    - Fail-open: if the endpoint is unreachable, we silently install a no-op
      exporter instead of raising errors or logging noisy failures.
    - Headers are parsed from the config's OTEL_EXPORTER_OTLP_HEADERS string.

Public API:
    init_tracer(): Initialize and return the tracer (idempotent).
    get_tracer(): Return an existing tracer, or initialize it if needed.
    force_flush(): Force flushing of any pending spans, if provider exists.
"""

from __future__ import annotations

from typing import Optional, Dict
from urllib.parse import urlparse
import socket

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from app.config import cfg

# Globals for the active tracer and provider instance
_tracer: Optional[trace.Tracer] = None
_provider: Optional[TracerProvider] = None


class _NoopExporter(SpanExporter):
    """Exporter that accepts spans and silently discards them.

    Used when tracing is disabled or when the configured OTLP endpoint is
    unreachable. Unlike the default no-op, this avoids logging errors or
    warnings when export is attempted.
    """

    def export(self, spans):  # type: ignore[override]
        return SpanExportResult.SUCCESS

    def shutdown(self):  # type: ignore[override]
        return None


def _parse_headers(h: str | None) -> Dict[str, str] | None:
    """Parse OTLP headers from a semicolon-separated string.

    Example:
        "api-key=123;env=prod" → {"api-key": "123", "env": "prod"}

    Args:
        h: Raw header string from configuration.

    Returns:
        A dictionary of parsed headers, or None if the input is empty.
    """
    if not h:
        return None
    parts = [p.strip() for p in h.split(";") if "=" in p]
    return {k.strip(): v.strip() for k, v in (p.split("=", 1) for p in parts)}


def _endpoint_reachable(url: str, timeout_s: float = 0.4) -> bool:
    """Best-effort TCP reachability check.

    Avoids noisy exporter errors by checking connectivity before wiring the
    exporter. Attempts to open a TCP socket to the host and port derived from
    the URL.

    Args:
        url: Endpoint URL string.
        timeout_s: Timeout in seconds for the connection attempt.

    Returns:
        True if a TCP connection could be established, else False.
    """
    try:
        u = urlparse(url)
        host = u.hostname
        if not host:
            return False
        port = u.port or (443 if u.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def init_tracer() -> trace.Tracer:
    """Initialize the global tracer.

    Behavior:
        - If already initialized, return the cached tracer.
        - If tracing is disabled, return a no-op tracer.
        - If enabled, create a TracerProvider with resource metadata, attach
          a BatchSpanProcessor, and wire an OTLP exporter if reachable.

    Returns:
        An OpenTelemetry Tracer instance.
    """
    global _tracer, _provider
    if _tracer:
        return _tracer

    # If disabled, return a no-op tracer (DIY JSON traces still work).
    if not cfg.TRACING_ENABLED:
        _tracer = trace.get_tracer(cfg.SERVICE_NAME)
        return _tracer

    resource = Resource.create(
        {
            "service.name": cfg.SERVICE_NAME,
            "service.version": "0.1.0",
            "deployment.environment": cfg.ENV,
        }
    )
    _provider = TracerProvider(resource=resource)

    # Fail-open: only wire the network exporter if the endpoint is reachable.
    exporter: SpanExporter
    if _endpoint_reachable(cfg.OTEL_EXPORTER_OTLP_ENDPOINT):
        exporter = OTLPSpanExporter(
            endpoint=cfg.OTEL_EXPORTER_OTLP_ENDPOINT,
            headers=_parse_headers(cfg.OTEL_EXPORTER_OTLP_HEADERS),
        )
    else:
        exporter = _NoopExporter()

    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer(cfg.SERVICE_NAME)
    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the active tracer, initializing if necessary."""
    return _tracer or init_tracer()


def force_flush(timeout_ms: int = 3000):
    """Force flush spans from the active provider.

    Args:
        timeout_ms: Maximum time to wait for flush, in milliseconds.
    """
    try:
        if _provider:
            _provider.force_flush(timeout_ms)
    except Exception:
        # Defensive: ignore errors to avoid blocking shutdown paths.
        pass
