# app/metrics.py
from prometheus_client import Counter, Histogram, start_http_server
from .config import cfg

_started = False

def _start_server_once():
    global _started
    if not _started and cfg.METRICS_ENABLED:
        try:
            start_http_server(cfg.METRICS_PORT)  # Local metrics server for Prometheus to scrape
            _started = True
        except OSError:
            # Already started (common on Streamlit reruns)
            _started = True

# Start the server at import time (safe due to guard)
_start_server_once()

REQS = Counter("rag_requests_total", "Total requests", ["route"])
ERRS = Counter("rag_errors_total", "Errors by type", ["route", "type"])
LAT_E2E = Histogram("rag_latency_seconds", "End-to-end latency", ["route"],
                    buckets=[0.2, 0.5, 1, 2, 3, 5, 8, 13])

# Optional; if you can compute token usage later, keep these:
TOK_IN = Histogram("rag_tokens_in", "Prompt tokens", ["model"], buckets=[128,256,512,1024,2048,4096])
TOK_OUT = Histogram("rag_tokens_out", "Completion tokens", ["model"], buckets=[64,128,256,512,1024])
