"""
log_broker.py – SSE broadcast: mọi module push event vào đây,
dashboard subscribe qua /api/log-stream.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Danh sách subscriber async queues
_subscribers: list[asyncio.Queue] = []


def push(
    message: str,
    level: str = "INFO",
    event_type: str | None = None,
    **extra: Any,
) -> None:
    """
    Push một log event đến tất cả subscriber đang listen.
    Có thể gọi từ bất kỳ thread đồng bộ nào (xử lý OpenCV, LLM...).
    """
    now = datetime.now().strftime("%H:%M:%S")
    payload: dict[str, Any] = {
        "time": now,
        "level": level,
        "message": message,
    }
    if event_type:
        payload["type"] = event_type
    payload.update(extra)

    data = json.dumps(payload, ensure_ascii=False)

    dead: list[asyncio.Queue] = []
    for q in _subscribers:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subscribers.remove(q)


def subscribe() -> asyncio.Queue:
    """Tạo một queue mới để nhận events, trả về queue đó."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """Xoá queue khi client disconnect."""
    if q in _subscribers:
        _subscribers.remove(q)


# ─── Python logging handler → SSE ────────────────────────────────────────────

class SSELogHandler(logging.Handler):
    """Chuyển Python log records → SSE push."""

    LEVEL_MAP = {
        logging.DEBUG:    "INFO",
        logging.INFO:     "INFO",
        logging.WARNING:  "WARN",
        logging.ERROR:    "ERR",
        logging.CRITICAL: "ERR",
    }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = self.LEVEL_MAP.get(record.levelno, "INFO")
            push(self.format(record), level=level)
        except Exception:
            pass


def install_log_handler(level: int = logging.INFO) -> None:
    """Cài SSELogHandler vào root logger."""
    handler = SSELogHandler()
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
