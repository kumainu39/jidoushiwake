from __future__ import annotations

import io
import os
import re
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: debug_print.py <path> <pattern> [context=3]")
        sys.exit(1)
    path = sys.argv[1]
    pattern = sys.argv[2]
    ctx = int(sys.argv[3]) if len(sys.argv) >= 4 else 3
    rx = re.compile(pattern)
    with io.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if rx.search(line):
            start = max(0, i - ctx)
            end = min(len(lines), i + ctx + 1)
            print(f"-- match at line {i+1}")
            for j in range(start, end):
                sys.stdout.write(f"{j+1:5d}: {lines[j]}")
            print()


if __name__ == '__main__':
    main()
