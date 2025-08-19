# app/tracing.py
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import os

from app.config import cfg

_tracer: Optional[trace.Tracer] = None
_provider: Optional[TracerProvider] = None

def _parse_headers(s: str | None):
    if not s:
        return None
    pairs = [p.strip() for p in s.split(";") if "=" in p]
    return {k.strip(): v.strip() for k, v in (p.split("=", 1) for p in pairs)}

def init_tracer() -> trace.Tracer:
    global _tracer, _provider
    if _tracer:
        return _tracer

    resource = Resource.create({
        "service.name": cfg.SERVICE_NAME,
        "service.version": "0.1.0",
        "deployment.environment": cfg.ENV,
    })

    # Local default: Jaeger all-in-one HTTP ingest
    endpoint = os.getenv("OTLP_HTTP_ENDPOINT", "http://localhost:4318/v1/traces")
    headers = _parse_headers(os.getenv("OTLP_HEADERS"))

    _provider = TracerProvider(resource=resource)
    _provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint, headers=headers)
    ))
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
