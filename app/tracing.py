# app/tracing.py
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

_tracer: Optional[trace.Tracer] = None
_provider: Optional[TracerProvider] = None


class _NoopExporter(SpanExporter):
    """Exporter that accepts spans and does nothing, without logging errors."""
    def export(self, spans):  # type: ignore[override]
        return SpanExportResult.SUCCESS

    def shutdown(self):  # type: ignore[override]
        return None


def _parse_headers(h: str | None) -> Dict[str, str] | None:
    if not h:
        return None
    parts = [p.strip() for p in h.split(";") if "=" in p]
    return {k.strip(): v.strip() for k, v in (p.split("=", 1) for p in parts)}


def _endpoint_reachable(url: str, timeout_s: float = 0.4) -> bool:
    """Best-effort TCP check; avoids noisy exporter errors if nothing is listening."""
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
    global _tracer, _provider
    if _tracer:
        return _tracer

    # If disabled, return a no-op tracer (your DIY JSON traces still work).
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
    return _tracer or init_tracer()


def force_flush(timeout_ms: int = 3000):
    try:
        if _provider:
            _provider.force_flush(timeout_ms)
    except Exception:
        pass
