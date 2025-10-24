from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    base_dir: Path
    data_dir: Path
    output_dir: Path
    logs_dir: Path
    database_url: str


def get_settings() -> Settings:
    base = Path(__file__).resolve().parents[2]
    data_dir = base / "data"
    output_dir = base / "output"
    logs_dir = base / "logs"

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # Default to SQLite in data dir
        data_dir.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{(data_dir / 'app.db').as_posix()}"

    return Settings(
        base_dir=base,
        data_dir=data_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        database_url=db_url,
    )

