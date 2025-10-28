from __future__ import annotations

from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.jidoushiwake.db import engine, Base, get_session  # type: ignore
from src.jidoushiwake.services import import_account_master_from_text  # type: ignore


def main() -> None:
    Base.metadata.create_all(engine)
    seed_path = Path('data/account_master_seed.txt')
    if not seed_path.exists():
        raise SystemExit(f'Seed file not found: {seed_path}')
    text = seed_path.read_text(encoding='utf-8', errors='ignore')
    with get_session() as s:
        n = import_account_master_from_text(s, text)
        print(f'Imported master accounts: {n}')


if __name__ == '__main__':
    main()
