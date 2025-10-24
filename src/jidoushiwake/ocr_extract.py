from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import os
import base64
import json


LOGGER = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    LOGGER.info("Extracting text from PDF: %s", pdf_path)
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore

        text = pdfminer_extract_text(str(pdf_path))
        if text and text.strip():
            LOGGER.info("Text extracted via pdfminer (%d chars)", len(text))
            return text
    except Exception as e:
        LOGGER.warning("pdfminer extract failed: %s", e)

    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
        text = "\n".join(texts)
        LOGGER.info("Text extracted via PyPDF2 (%d chars)", len(text))
        return text
    except Exception as e:
        LOGGER.error("PyPDF2 extract failed: %s", e)
        return ""


def _extract_text_with_paddle(pdf_path: Path) -> str:
    """Best-effort OCR using PaddleOCR by rasterizing PDF pages.

    Returns empty string if PaddleOCR or rasterizer is unavailable.
    """
    try:
        # Lazy imports to avoid hard dependency when not installed
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as e:
        LOGGER.debug("PaddleOCR not available: %s", e)
        return ""

    # Render PDF pages to images using PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        LOGGER.warning("PyMuPDF not available for rasterization: %s", e)
        return ""

    try:
        # Japanese OCR
        ocr = PaddleOCR(lang="japan", show_log=False)
    except Exception as e:
        LOGGER.error("Failed to init PaddleOCR: %s", e)
        return ""

    texts: list[str] = []
    try:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            # 2x zoom for better OCR
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            # Convert to bytes (PNG)
            img_bytes = pix.tobytes("png")
            # PaddleOCR can take image path or ndarray; use ndarray via cv2.imdecode
            try:
                import numpy as np  # type: ignore
                import cv2  # type: ignore

                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                result = ocr.ocr(img, cls=True)
            except Exception:
                # Fallback: write to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
                    tf.write(img_bytes)
                    tf.flush()
                    result = ocr.ocr(tf.name, cls=True)

            for line in result or []:
                for det in line:
                    try:
                        txt = det[1][0]
                    except Exception:
                        continue
                    if txt:
                        texts.append(str(txt))
    except Exception as e:
        LOGGER.error("PaddleOCR processing failed: %s", e)
        return ""

    return "\n".join(texts)


def _extract_text_with_yomitoku(pdf_path: Path) -> str:
    """Best-effort OCR via YOMITOKU CLI (not a Python library).

    Expects `yomitoku` command to be available in PATH. Mirrors the
    typical usage:
        yomitoku <pdf> -f md --combine --encoding utf-8 -o <outdir>

    Returns concatenated text from generated files (preferring .md).
    Returns empty string if CLI is not available or on failure.
    """
    try:
        import shutil
        import subprocess
        import tempfile
    except Exception:
        return ""

    if not shutil.which("yomitoku"):
        return ""

    try:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "yomitoku_output"
            out_dir.mkdir(parents=True, exist_ok=True)
            fmt = os.getenv("YOMITOKU_FORMAT", "md")
            cmd = [
                "yomitoku",
                str(pdf_path),
                "-f", fmt,
                "--combine",
                "--encoding", "utf-8",
                "-o", str(out_dir),
            ]
            try:
                subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                return ""

            # Prefer reading .md files, then .txt, then any text-like files
            collected: list[str] = []
            patterns = ["**/*.md", "**/*.txt", "**/*.json", "**/*.*"]
            for pat in patterns:
                for f in sorted(out_dir.glob(pat)):
                    if f.is_dir():
                        continue
                    try:
                        txt = f.read_text(encoding="utf-8", errors="ignore")
                        if txt and txt.strip():
                            collected.append(txt)
                    except Exception:
                        continue
                if collected:
                    break
            return "\n\n".join(collected)
    except Exception:
        return ""


def extract_text_both(pdf_path: Path) -> dict:
    """Extract text via embedded text layer, YOMITOKU, and PaddleOCR.

    Returns a dict with keys: text_pdf, text_yomitoku, text_paddle, text_combined.
    """
    t_pdf = extract_text_from_pdf(pdf_path)
    t_yomi = _extract_text_with_yomitoku(pdf_path)
    t_paddle = _extract_text_with_paddle(pdf_path)

    # Combine heuristically: merge unique lines preserving order preference to PDF text
    seen = set()
    combined_lines: list[str] = []
    # Prefer YOMITOKU > PDF layer > PaddleOCR
    for part in (t_yomi, t_pdf, t_paddle):
        for ln in (part or "").splitlines():
            s = ln.strip()
            if not s:
                continue
            key = s
            if key in seen:
                continue
            seen.add(key)
            combined_lines.append(s)

    return {
        "text_pdf": t_pdf,
        "text_yomitoku": t_yomi,
        "text_paddle": t_paddle,
        "text_combined": "\n".join(combined_lines),
    }


@dataclass
class ParsedJournal:
    date: Optional[str]
    amount: Optional[int]
    summary: str
    debit_account: str
    credit_account: str
    counterparty: str = ""


DATE_PATTERNS = [
    re.compile(r"(20\d{2})[\-/\.](\d{1,2})[\-/\.](\d{1,2})"),
    re.compile(r"(\d{4})(\d{2})(\d{2})"),
]

YEN_AMOUNT_PATTERNS = [
    re.compile(r"([\-−]?)\s*¥?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*(?:円)?"),
]


def _norm_date(y: str, m: str, d: str) -> Optional[str]:
    try:
        dt = datetime(int(y), int(m), int(d))
        return dt.strftime("%Y/%m/%d")
    except ValueError:
        return None


def _find_date(text: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if pat.groups == 3 and len(m.groups()) == 3:
            y, a, b = m.groups()
            return _norm_date(y, a, b)
    return None


def _parse_int_amount(s: str, sign: str = "") -> int:
    s = s.replace(",", "")
    try:
        val = int(s)
        return -val if sign in ("-", "−") else val
    except ValueError:
        return 0


def _find_amount(text: str) -> Optional[int]:
    candidates: list[tuple[int, int]] = []
    for line in text.splitlines():
        lowered = line.lower()
        weight = 1
        if any(k in lowered for k in ("合計", "計", "金額", "精算", "請求額", "税込", "支払")):
            weight = 3
        for pat in YEN_AMOUNT_PATTERNS:
            for m in pat.finditer(line):
                sign = m.group(1) or ""
                num = m.group(2)
                amount = _parse_int_amount(num, sign)
                if amount == 0:
                    continue
                candidates.append((weight * abs(amount), amount))
    if not candidates:
        return None
    _, best_amount = max(candidates, key=lambda t: t[0])
    return best_amount


VENDOR_KEYWORDS = {
    "amazon": "消耗品費",
    "アマゾン": "消耗品費",
    "楽天": "消耗品費",
    "ヤマト": "荷造運賃",
    "佐川": "荷造運賃",
    "ゆうパック": "荷造運賃",
    "郵便": "通信費",
    "切手": "通信費",
    "タクシー": "旅費交通費",
    "uber": "旅費交通費",
    "jr": "旅費交通費",
    "電車": "旅費交通費",
    "ガソリン": "車両費",
    "eneos": "車両費",
    "出前館": "会議費",
    "ウーバーイーツ": "会議費",
    "マクドナルド": "会議費",
    "スターバックス": "交際費",
}

PAYMENT_KEYWORDS = {
    "クレジット": "未払金",
    "visa": "未払金",
    "mastercard": "未払金",
    "jcb": "未払金",
    "amex": "未払金",
    "paypay": "未払金",
    "line pay": "未払金",
    "楽天ペイ": "未払金",
    "請求書": "未払金",
    "振込": "普通預金",
    "振替": "普通預金",
    "入金": "普通預金",
    "引落": "普通預金",
    "現金": "現金",
    "レジ": "現金",
    "atm": "普通預金",
    "銀行": "普通預金",
}


def _guess_accounts(text: str) -> Tuple[str, str]:
    t = text.lower()
    for kw, acc in VENDOR_KEYWORDS.items():
        if kw in t:
            debit = acc
            break
    else:
        debit = "雑費"

    for kw, acc in PAYMENT_KEYWORDS.items():
        if kw in t:
            credit = acc
            break
    else:
        credit = "未払金"
    return debit, credit


def _extract_counterparty(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = lines[:25]
    best = ""
    for ln in head:
        if len(ln) < 3:
            continue
        if re.search(r"[A-Za-zァ-ンｧ-ﾝﾞﾟ一-龥]{2,}", ln):
            if any(k in ln for k in ("領収", "請求", "合計", "金額", "明細", "内訳")):
                continue
            best = ln
            break
    return best[:64]


def extract_journal_data(text: str) -> ParsedJournal:
    date = _find_date(text)
    amount = _find_amount(text)
    debit, credit = _guess_accounts(text)
    counterparty = _extract_counterparty(text)
    summary_parts = [p for p in [counterparty or None, "購入"] if p]
    summary = " ".join(summary_parts) if summary_parts else "支払"
    return ParsedJournal(
        date=date,
        amount=amount,
        summary=summary,
        debit_account=debit,
        credit_account=credit,
        counterparty=counterparty,
    )
