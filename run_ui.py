from __future__ import annotations

import threading

import uvicorn

from src.jidoushiwake.api.app import app
from src.jidoushiwake.frontend.app import run_ui


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
    t = threading.Thread(target=start_api, daemon=True)
    t.start()
    run_ui()
