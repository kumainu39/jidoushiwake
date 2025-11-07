from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Any, List, Dict, Callable


LOGGER = logging.getLogger(__name__)

def _log_progress(
    msg: str,
    progress: Optional[Callable[[str], None]] = None,
    run_log: Optional[Path] = None,
) -> None:
    try:
        LOGGER.info(msg)
    except Exception:
        pass
    try:
        if progress:
            progress(msg)
            return
        # fallback: write to logs/yomitoku.log or provided run_log
        try:
            if run_log is None:
                from .config import get_settings  # local import to avoid hard dependency at import time

                log_dir = get_settings().logs_dir
                log_dir.mkdir(parents=True, exist_ok=True)
                lf = log_dir / "yomitoku.log"
            else:
                run_log.parent.mkdir(parents=True, exist_ok=True)
                lf = run_log
            lf.open("a", encoding="utf-8").write(msg + "\n")
        except Exception:
            pass
    except Exception:
        pass


def _resolve_yomitoku_exe() -> Optional[str]:
    try:
        import shutil
    except Exception:
        return None

    p = shutil.which("yomitoku")
    if p:
        return p

    try:
        here = Path(sys.executable).parent
        for name in ("yomitoku.exe", "yomitoku"):
            cand = here / name
            if cand.exists():
                return str(cand)
    except Exception:
        pass

    for rel in (".venv/Scripts/yomitoku.exe", ".venv/bin/yomitoku", "venv/Scripts/yomitoku.exe", "venv/bin/yomitoku"):
        cand = Path.cwd() / rel
        if cand.exists():
            return str(cand)

    env_p = os.getenv("YOMITOKU_EXE")
    if env_p and Path(env_p).exists():
        return env_p

    return None


def _coerce_to_text(value: Any) -> str:
    """Best-effort conversion of various return types to text."""
    try:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        if isinstance(value, dict):
            t = value.get("text")
            if isinstance(t, (str, bytes)):
                return _coerce_to_text(t)
            parts: list[str] = []
            for v in value.values():
                s = _coerce_to_text(v)
                if s:
                    parts.append(s)
            return "\n\n".join(parts)
        if isinstance(value, (list, tuple)):
            parts = [s for s in (_coerce_to_text(v) for v in value) if s]
            return "\n".join(parts)
    except Exception:
        return ""
    return ""


def extract_text_with_yomitoku(
    pdf_path: Path,
    ensure: bool = False,
    output_dir: Path | None = None,
    fmt: str | None = None,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Extract text from PDF using YOMITOKU.

    Order:
    1) Try Python module import patterns if installed
    2) Fallback to CLI `yomitoku`

    - When `ensure=True`, raise RuntimeError if YOMITOKU cannot be used or returns empty text.
    - `output_dir` controls where CLI output files are written (defaults to `<pdf_dir>/yomitoku_output` or env `YOMITOKU_OUTPUT_DIR`).
    - `fmt` overrides output format (defaults to env `YOMITOKU_FORMAT` or `md`).
    """

    # Prepare per-run log file in logs dir
    run_log: Optional[Path] = None
    try:
        from .config import get_settings  # type: ignore

        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log = get_settings().logs_dir / f"yomitoku_run_{ts}.log"
    except Exception:
        run_log = None

    _log_progress(f"[YOMITOKU] start: {pdf_path}", progress, run_log)
    if not isinstance(pdf_path, Path):
        pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        msg = f"PDF not found: {pdf_path}"
        if ensure:
            raise RuntimeError(msg)
        _log_progress(f"[YOMITOKU] {msg}", progress, run_log)
        return ""

    text: str = ""

    # 1) Python package best-effort
    try:
        try:
            import yomitoku as _yomi  # type: ignore
        except Exception:
            _yomi = None  # type: ignore
        if _yomi is not None:
            _log_progress("[YOMITOKU] using Python package", progress, run_log)
            for attr in ("extract_text", "convert", "to_text", "extract"):
                func = getattr(_yomi, attr, None)
                if callable(func):
                    try:
                        t = func(str(pdf_path))  # type: ignore[misc]
                        t_str = _coerce_to_text(t)
                        if t_str and t_str.strip():
                            text = t_str
                            _log_progress(f"[YOMITOKU] package {attr} -> {len(text)} chars", progress, run_log)
                            break
                    except Exception:
                        continue
            if not text:
                api = getattr(_yomi, "api", None)
                if api is not None:
                    for attr in ("extract_text", "convert", "to_text"):
                        func = getattr(api, attr, None)
                        if callable(func):
                            try:
                                t = func(str(pdf_path))  # type: ignore[misc]
                                t_str = _coerce_to_text(t)
                                if t_str and t_str.strip():
                                    text = t_str
                                    _log_progress(f"[YOMITOKU] package api.{attr} -> {len(text)} chars", progress, run_log)
                                    break
                            except Exception:
                                continue
    except Exception:
        # Continue to CLI fallback
        pass

    # 2) CLI fallback if package did not yield text
    if not text:
        exe = _resolve_yomitoku_exe()
        if exe is not None:
            try:
                base_dir = pdf_path.parent
                base_out = output_dir or (
                    Path(os.getenv("YOMITOKU_OUTPUT_DIR"))
                    if os.getenv("YOMITOKU_OUTPUT_DIR")
                    else (base_dir / "yomitoku_output")
                )
                base_out.mkdir(parents=True, exist_ok=True)

                # Use a per-run subdirectory to avoid collisions between files.
                # If filename has non-ASCII chars, also include a timestamp for uniqueness.
                try:
                    import time, re, random

                    stem = pdf_path.stem
                    stem_ascii = re.sub(r"[^0-9A-Za-z._-]", "_", stem)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    run_out = base_out / f"{stem_ascii}_{ts}_{random.randint(1000,9999)}"
                except Exception:
                    run_out = base_out / "run"
                run_out.mkdir(parents=True, exist_ok=True)
                fmt_val = fmt or os.getenv("YOMITOKU_FORMAT", "md")

                # Workaround: Some environments fail with non-ASCII filenames. Copy to temp with ASCII name.
                pdf_for_cli = pdf_path
                try:
                    ascii_ok = all(ord(c) < 128 for c in str(pdf_path))
                    if not ascii_ok or any(ch in str(pdf_path) for ch in (' ', '%', '(', ')', '　', '％', '（', '）')):
                        import tempfile, shutil

                        tmpdir = Path(tempfile.mkdtemp(prefix="yomi_cli_", dir=str(out_dir)))
                        pdf_for_cli = tmpdir / "input.pdf"
                        shutil.copy2(str(pdf_path), str(pdf_for_cli))
                        _log_progress(f"[YOMITOKU] copied to temp ASCII path: {pdf_for_cli}", progress, run_log)
                except Exception:
                    pdf_for_cli = pdf_path

                cmd = [
                    exe,
                    str(pdf_for_cli),
                    "-f",
                    fmt_val,
                    "--combine",
                    "--encoding",
                    "utf-8",
                    "-o",
                    str(run_out),
                ]

                # Optional extra args via env (e.g., "--lang ja --dpi 300")
                try:
                    extra = os.getenv("YOMITOKU_EXTRA_ARGS")
                    if extra:
                        import shlex

                        cmd.extend(shlex.split(extra, posix=False))
                        _log_progress(f"[YOMITOKU] ExtraArgs: {extra}", progress, run_log)
                except Exception:
                    pass
                _log_progress(f"[YOMITOKU] CLI: {exe}", progress, run_log)
                _log_progress(f"[YOMITOKU] OutDir: {run_out}", progress, run_log)
                _log_progress(f"[YOMITOKU] Cmd: {' '.join(cmd)}", progress, run_log)

                timeout_env = os.getenv("YOMITOKU_TIMEOUT")
                try:
                    timeout_s = int(timeout_env) if timeout_env else 180
                except Exception:
                    timeout_s = 180
                timeout_s = max(10, min(timeout_s, 1800))

                env = os.environ.copy()
                env.setdefault("PYTHONUTF8", "1")
                env.setdefault("PYTHONIOENCODING", "utf-8")
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=timeout_s,
                    env=env,
                )
                _log_progress(f"[YOMITOKU] CLI returncode: {proc.returncode}", progress, run_log)
                if proc.returncode != 0:
                    _log_progress(f"[YOMITOKU] CLI stderr: {(proc.stderr or '').strip()[:2000]}", progress, run_log)
                # Always store full stdout/stderr to the run log for debugging
                try:
                    if run_log is not None:
                        with run_log.open("a", encoding="utf-8") as f:
                            f.write("\n----- CLI STDOUT -----\n")
                            f.write((proc.stdout or "") + "\n")
                            f.write("\n----- CLI STDERR -----\n")
                            f.write((proc.stderr or "") + "\n")
                except Exception:
                    pass

                collected: list[str] = []
                # Prefer typical combined file names first
                preferred = [
                    "combined.md",
                    "combined.txt",
                    "combined.json",
                ]
                for name in preferred:
                    f = run_out / name
                    if f.exists() and f.is_file():
                        try:
                            txt = f.read_text(encoding="utf-8", errors="ignore")
                            if txt and txt.strip():
                                collected.append(txt)
                                _log_progress(f"[YOMITOKU] collected preferred: {name} ({len(txt)} chars)", progress, run_log)
                                break
                        except Exception:
                            pass
                if not collected:
                    for pat in ("**/*.md", "**/*.txt", "**/*.json", "**/*.*"):
                        for f in sorted(run_out.glob(pat)):
                            if f.is_dir():
                                continue
                            try:
                                txt = f.read_text(encoding="utf-8", errors="ignore")
                                if txt and txt.strip():
                                    collected.append(txt)
                                    _log_progress(f"[YOMITOKU] collected: {f.name} ({len(txt)} chars)", progress, run_log)
                            except Exception:
                                continue
                        if collected:
                            break
                text = "\n\n".join(collected)
                _log_progress(f"[YOMITOKU] total collected text: {len(text)} chars", progress, run_log)
            except Exception as e:
                _log_progress(f"[YOMITOKU] CLI invocation failed: {e}", progress, run_log)
                text = ""
        else:
            if ensure:
                raise RuntimeError("YOMITOKU not usable: CLI not found and package import failed")

    if ensure and not (text and text.strip()):
        where = f" (log: {run_log})" if run_log is not None else ""
        raise RuntimeError(
            f"YOMITOKU OCR produced no text for: {pdf_path}.{where} "
            "Ensure YOMITOKU Python package or CLI works. "
            "Set YOMITOKU_EXE to the CLI path if needed."
        )

    _log_progress("[YOMITOKU] done", progress, run_log)
    return text


def is_yomitoku_available() -> bool:
    """Quick check to see if YOMITOKU is likely usable."""
    try:
        import importlib

        if importlib.util.find_spec("yomitoku") is not None:  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    return _resolve_yomitoku_exe() is not None


# Optional: retain a simple CLI entry for manual debugging of YOMITOKU behavior
def _collect_dir_files(directory: Path, pattern: str = "*") -> List[Path]:
    return sorted([f for f in directory.glob(pattern) if f.is_file()])


def process_pdf(
    pdf_path: str | os.PathLike[str],
    *,
    output_format: str = "md",
    combine: bool = True,
    encoding: str = "utf-8",
    output_dir: str | os.PathLike[str] | None = None,
) -> Dict[str, object]:
    pdf = Path(pdf_path)
    if not pdf.exists():
        return {"ok": False, "returncode": -2, "output_dir": str(output_dir or ""), "output_files": [], "stdout": "", "stderr": f"not found: {pdf}"}

    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (pdf.parent / "yomitoku_output")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    before = set(_collect_dir_files(out_dir, pattern=f"*.{output_format}"))

    exe = _resolve_yomitoku_exe() or "yomitoku"
    cmd: List[str] = [
        exe,
        str(pdf),
        "-f",
        output_format,
        "--encoding",
        encoding,
        "-o",
        str(out_dir),
    ]
    if combine:
        cmd.insert(4, "--combine")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    except FileNotFoundError as e:
        return {
            "ok": False,
            "returncode": -1,
            "output_dir": str(out_dir),
            "output_files": [],
            "stdout": "",
            "stderr": f"yomitoku not found: {e}",
        }

    after = set(_collect_dir_files(out_dir, pattern=f"*.{output_format}"))
    new_files = [str(p) for p in sorted(after - before)]

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "output_dir": str(out_dir),
        "output_files": new_files,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
