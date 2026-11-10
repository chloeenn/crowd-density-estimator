"""Structured JSON logs. Only whitelisted fields are emitted, so an image byte or an upload
filename can never leak into a log line by accident."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

LOGGER_NAME = "crowd.api"
LOG_FIELDS = (
    "request_id",
    "method",
    "path",
    "status",
    "duration_ms",
    "inference_ms",
    "part",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        out |= {k: getattr(record, k) for k in LOG_FIELDS if hasattr(record, k)}
        if record.exc_info:  # server-side only; clients get a generic 500
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def setup_logging(level: str) -> logging.Logger:
    """Idempotent: replaces this logger's handlers, so repeated app creation can't duplicate."""
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers[:] = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger
