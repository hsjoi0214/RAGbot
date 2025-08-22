from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Data & index
    TEXT_FILE: str = os.getenv("TEXT_FILE", "data/crime_and_punishment.txt")
    PERSIST_DIR: str = os.getenv("PERSIST_DIR", "storage/vector_index")

    # Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Observability
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "my-app")
    TRACING_ENABLED: bool = os.getenv("TRACING_ENABLED", "1") == "1"
    # Replace Grafana Cloud with local OTLP endpoint for testing
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:14250")  # Local Jaeger or OTLP endpoint
    OTEL_EXPORTER_OTLP_HEADERS: str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "1") == "1"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9000"))

    # Misc
    ENV: str = os.getenv("ENV", "production") 

cfg = Config()

# Create folders that must exist
Path("data").mkdir(exist_ok=True, parents=True)
Path(cfg.PERSIST_DIR).mkdir(exist_ok=True, parents=True)
