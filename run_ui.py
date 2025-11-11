from __future__ import annotations

import threading
from pathlib import Path

import uvicorn

from src.jidoushiwake.api.app import app
# ユーザー提供の最新UI（app.py）を使用
from src.jidoushiwake.frontend.app import run_ui
from src.jidoushiwake.logging_utils import setup_logging


def start_api() -> None:
    host = "127.0.0.1"
    base_port = 8765
    # Try a small range of ports to avoid EADDRINUSE
    for p in range(base_port, base_port + 10):
        try:
            uvicorn.run(app, host=host, port=p, log_level="info")
            break
        except OSError:
            continue


if __name__ == "__main__":
    # Enable both console and file logging for terminal visibility
    setup_logging(Path("logs") / "app.log")

    t = threading.Thread(target=start_api, daemon=True)
    t.start()
    run_ui()
