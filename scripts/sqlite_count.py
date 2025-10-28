from pathlib import Path
import sqlite3

db = Path(__file__).resolve().parents[1] / 'data' / 'app.db'
conn = sqlite3.connect(db)
cur = conn.cursor()
try:
    cur.execute('select count(*) from account_master')
    print(cur.fetchone()[0])
except Exception as e:
    print('ERR', e)
