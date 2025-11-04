from __future__ import annotations

import io
import sys


def main() -> None:
    path = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    with io.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, l in enumerate(f, 1):
            if start <= i <= end:
                sys.stdout.write(f"{i:5d}: {l}")
            if i > end:
                break


if __name__ == '__main__':
    main()

