from __future__ import annotations

from pathlib import Path
import sqlite3
import re


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    db_path = base / 'data' / 'app.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS account_master ('
        'id INTEGER PRIMARY KEY AUTOINCREMENT,'
        'code VARCHAR(32),'
        'name VARCHAR(128) UNIQUE,'
        'category VARCHAR(64),'
        'created_at TEXT DEFAULT CURRENT_TIMESTAMP,'
        'updated_at TEXT DEFAULT CURRENT_TIMESTAMP)'
    )
    seed = (base / 'data' / 'account_master_seed.txt').read_text(encoding='utf-8', errors='ignore')
    skip_kw = (
        '資産の部','流動資産','固定資産','繰延資産','諸口',
        '負債の部','流動負債','固定負債','純資産の部','収益の部',
        '売上高','売上原価','販売費および一般管理費','営業外','特別損益','など'
    )
    ins = 0
    debug_miss = 0
    for ln in seed.splitlines():
        ln = ln.strip()
        if not ln or any(k in ln for k in skip_kw):
            continue
        # capture name before first uppercase code token
        m = re.match(r'^(.+?)\s+[A-Z][A-Z0-9\-]+(?:\s+\d+)?\s*$', ln)
        if not m:
            debug_miss += 1
            continue
        name = m.group(1).strip()
        # try to capture code if present
        m2 = re.search(r'\s([A-Z][A-Z0-9\-]+)(?:\s+\d+)?\s*$', ln)
        code = m2.group(1) if m2 else None
        cur.execute('INSERT OR IGNORE INTO account_master(name, code, category) VALUES (?,?,?)', (name, code, None))
        if cur.rowcount:
            ins += 1
    conn.commit()
    print(f'Imported to account_master: {ins} (skipped {debug_miss})')


if __name__ == '__main__':
    main()
