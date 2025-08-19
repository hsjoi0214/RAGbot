# app/tracing.py
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from app.config import cfg

_tracer_cached: Optional[trace.Tracer] = None
_initialized = False

def init_tracer() -> trace.Tracer:
    """
    Initialize OpenTelemetry once. Safe for Streamlit (idempotent).
    """
    global _initialized, _tracer_cached
    if _initialized and _tracer_cached:
        return _tracer_cached

    resource = Resource.create({"service.name": cfg.SERVICE_NAME, "service.version": "0.1.0"})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=cfg.JAEGER_OTLP_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer_cached = trace.get_tracer(cfg.SERVICE_NAME)
    _initialized = True
    return _tracer_cached

def get_tracer() -> trace.Tracer:
    return _tracer_cached or init_tracer()
