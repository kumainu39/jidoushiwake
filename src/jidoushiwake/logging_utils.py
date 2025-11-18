from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[Path | str] = None,
    level: int = logging.INFO,
    max_bytes: int = 1_000_000,
    backup_count: int = 5,
) -> None:
    """Configure root logger for both file and console.

    - Adds a RotatingFileHandler if `log_file` is provided
    - Always adds a StreamHandler to stderr
    - Idempotent: won't duplicate handlers on repeated calls
    """
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler if missing
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # Add rotating file handler if requested and missing
    if log_file:
        p = Path(log_file)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', '') == str(p) for h in root.handlers):
            fh = RotatingFileHandler(str(p), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            root.addHandler(fh)

