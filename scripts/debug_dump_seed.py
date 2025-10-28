from pathlib import Path
seed = Path('data/account_master_seed.txt').read_text(encoding='utf-8', errors='ignore')
for i,ln in enumerate(seed.splitlines(),1):
    if i>40: break
    print(i, repr(ln))
