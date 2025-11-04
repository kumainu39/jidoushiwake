from __future__ import annotations

import io
import sys


def main() -> None:
    p = sys.argv[1]
    needle = sys.argv[2]
    with io.open(p, 'r', encoding='utf-8', errors='ignore') as f:
        s = f.read()
    i = s.find(needle)
    if i < 0:
        print('NOT_FOUND')
        return
    start = max(0, i - 200)
    end = min(len(s), i + 800)
    print(s[start:end])


if __name__ == '__main__':
    main()

