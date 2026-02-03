"""
MOMENTUM-X Metrics HTTP Server

### ARCHITECTURAL CONTEXT
Node ID: monitoring.server
Graph Link: docs/memory/graph_state.json → "monitoring.server"

### RESEARCH BASIS
Exposes MetricsRegistry as HTTP endpoints for Prometheus scraping
and operational health checks.

Endpoints:
  GET /metrics  → Prometheus text exposition format
  GET /health   → JSON health check
  GET /snapshot  → JSON metrics snapshot

Uses stdlib http.server — no external dependencies.

Ref: ADR-018 (Observability)

### CRITICAL INVARIANTS
1. Non-blocking: runs in a daemon thread, never blocks trading loop.
2. Read-only: never modifies metrics, only reads.
3. Minimal overhead: no serialization until request arrives.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from src.monitoring.metrics import get_metrics

logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for metrics endpoints."""

    def do_GET(self) -> None:
        if self.path == "/metrics":
            self._serve_prometheus()
        elif self.path == "/health":
            self._serve_health()
        elif self.path == "/snapshot":
            self._serve_snapshot()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _serve_prometheus(self) -> None:
        metrics = get_metrics()
        body = metrics.to_prometheus().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_health(self) -> None:
        health = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tests_total": get_metrics().scan_iterations.value,
        }
        body = json.dumps(health).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_snapshot(self) -> None:
        metrics = get_metrics()
        body = json.dumps(metrics.snapshot(), indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging."""
        pass


class MetricsServer:
    """
    Lightweight HTTP server for metrics exposition.

    Node ID: monitoring.server
    Ref: ADR-018 (Observability)

    Usage:
        server = MetricsServer(port=9090)
        server.start()  # Runs in daemon thread
        # ... trading loop ...
        server.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9090) -> None:
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the metrics server in a daemon thread."""
        self._server = HTTPServer((self._host, self._port), MetricsHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="metrics-server",
        )
        self._thread.start()
        logger.info(
            "Metrics server started at http://%s:%d/metrics",
            self._host, self._port,
        )

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("Metrics server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def port(self) -> int:
        return self._port
