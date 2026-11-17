"""Structured JSON logging. Secrets are never passed to loggers; a filter scrubs
anything that looks like a key as defence in depth."""
from __future__ import annotations

import json
import logging
import re
import sys

_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|(?i:api[_-]?key)\s*[=:]\s*\S+)")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = _SECRET_RE.sub("<redacted>", record.getMessage())
        out = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name, "msg": msg}
        if hasattr(record, "extra_fields"):
            out.update(record.extra_fields)  # type: ignore[attr-defined]
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger("orchidshield")
    if root.handlers:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_JsonFormatter())
    root.addHandler(h)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"orchidshield.{name}")
