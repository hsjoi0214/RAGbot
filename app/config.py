"""
Application configuration module.

This module centralizes environment-driven configuration for the application.
It defines the `Config` dataclass, which provides immutable settings for data
storage, model selection, and observability tooling. Values are sourced from
environment variables with safe defaults, enabling twelve-factor style
configuration.

Side effects:
    - Ensures the `data/` directory exists.
    - Ensures the persistent storage directory (vector index) exists.

Usage:
    from config import cfg
    print(cfg.EMBEDDING_MODEL)
"""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Immutable application configuration.

    All attributes are derived from environment variables with defaults.
    This class is frozen (read-only) to prevent runtime mutation of config.

    Attributes:
        TEXT_FILE: Path to the source text file for processing.
        PERSIST_DIR: Directory for storing the vector index.
        EMBEDDING_MODEL: HuggingFace or sentence-transformers embedding model.
        GROQ_MODEL: LLM identifier for Groq inference.
        SERVICE_NAME: Logical name for observability/tracing backends.
        TRACING_ENABLED: Whether OpenTelemetry tracing is enabled.
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP/Jaeger endpoint for trace export.
        OTEL_EXPORTER_OTLP_HEADERS: Optional OTLP auth headers.
        METRICS_ENABLED: Whether Prometheus metrics export is enabled.
        METRICS_PORT: Port on which metrics will be served.
        ENV: Environment profile (e.g., production, staging, local).
    """

    # Data & index configuration
    TEXT_FILE: str = os.getenv("TEXT_FILE", "data/crime_and_punishment.txt")
    PERSIST_DIR: str = os.getenv("PERSIST_DIR", "storage/vector_index")

    # Model configuration
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Observability / telemetry
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "my-ragbot-app")
    TRACING_ENABLED: bool = os.getenv("TRACING_ENABLED", "1") == "1"
    # Default points to local Jaeger or OTLP endpoint for local development
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:14250"
    )
    OTEL_EXPORTER_OTLP_HEADERS: str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "1") == "1"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9000"))

    # Environment mode
    ENV: str = os.getenv("ENV", "production")


# Global config instance for application use
cfg = Config()

# Ensure required directories exist at import time.
# This prevents runtime errors later when reading/writing files.
Path("data").mkdir(exist_ok=True, parents=True)
Path(cfg.PERSIST_DIR).mkdir(exist_ok=True, parents=True)
