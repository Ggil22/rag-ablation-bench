"""Request tracing: one JSON line per request, and Langfuse traces when configured.

Every request is logged locally with its timings, token usage, retrieved chunk ids and the
answer, so latency percentiles and cost can be computed from the log alone. If
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (and LANGFUSE_HOST for a self-hosted
instance), each request also becomes a Langfuse trace with a retriever span and a
generation, opened around the real work so their durations are the real ones.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("rab.serve")


class _NoSpan:
    def update(self, **kwargs) -> None:
        pass


class Tracer:
    def __init__(self, log_path: Path | None, langfuse=None):
        self.log_path = log_path
        self.langfuse = langfuse
        self._lock = threading.Lock()
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Tracer:
        langfuse = None
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse

                langfuse = Langfuse()
            except ImportError:
                log.warning("Langfuse keys set but the langfuse package is not installed")
        path = os.environ.get("RAB_REQUEST_LOG", "logs/requests.jsonl")
        return cls(Path(path) if path else None, langfuse)

    @contextmanager
    def observe(self, name: str, *, as_type: str = "span", **kwargs):
        """A Langfuse observation around the block, or a no-op without Langfuse. Nested
        calls become children of the enclosing observation."""
        if self.langfuse is None:
            yield _NoSpan()
            return
        try:
            cm = self.langfuse.start_as_current_observation(name=name, as_type=as_type, **kwargs)
            span = cm.__enter__()
        except Exception:  # observability must never break a request
            log.exception("failed to start Langfuse observation")
            yield _NoSpan()
            return
        try:
            yield span
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                log.exception("failed to end Langfuse observation")

    def log_request(self, row: dict) -> None:
        if self.log_path is None:
            return
        row = {"ts": datetime.now(UTC).isoformat(), **row}
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        if self.langfuse is not None:
            self.langfuse.flush()
