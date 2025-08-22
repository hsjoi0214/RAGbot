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

    # Create the resource using the environment variables (loaded via the .env file)
    resource = Resource.create({
        "service.name": cfg.SERVICE_NAME,  # Ensure the service name is set from your config (like "rag-streamlit")
        "service.version": "0.1.0",  # You can update this if you use versioning
        "deployment.environment": cfg.ENV,  # Environment (e.g., "cloud")
    })

    # Use local OTLP or Jaeger endpoint for local testing
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:14250")  # Local Jaeger endpoint
    
    # Parse OTLP headers from the .env file
    headers = _parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS"))

    # Initialize OpenTelemetry's TracerProvider with the resource and the exporter (OTLP)
    _provider = TracerProvider(resource=resource)
    _provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint, headers=headers)
    ))

    # Set the tracer provider globally
    trace.set_tracer_provider(_provider)
    
    # Initialize the tracer
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
