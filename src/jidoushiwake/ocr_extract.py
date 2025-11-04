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

from .yomitoku_ocr import extract_text_with_yomitoku

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


# YOMITOKU OCR is now provided by a dedicated module for maintainability.


def extract_text_both(pdf_path: Path) -> dict:
    """Extract text via embedded text layer and YOMITOKU (PaddleOCR removed).

    Returns a dict with keys: text_pdf, text_yomitoku, text_combined.

    text_combined always uses YOMITOKU output (hard requirement).
    """
    LOGGER.info("[OCR] begin extract_text_both: %s", pdf_path)
    t_pdf = extract_text_from_pdf(pdf_path)
    LOGGER.info("[OCR] pdfminer/PyPDF2 extracted: %d chars", len(t_pdf or ""))
    t_yomi = extract_text_with_yomitoku(pdf_path, ensure=True)
    LOGGER.info("[OCR] YOMITOKU extracted: %d chars", len(t_yomi or ""))
    # Enforce YOMITOKU usage: if unavailable or empty, treat as fatal
    if not (t_yomi and t_yomi.strip()):
        raise RuntimeError(
            "YOMITOKU OCR is required but not available or produced no text. "
            "Install and ensure 'yomitoku' CLI (or Python package) works. "
            "You can set YOMITOKU_EXE to the CLI path."
        )
    # PaddleOCR usage removed by request
    t_paddle = ""

    # Prefer a single-source combined text: YOMITOKU > PDF > Paddle
    # Always use YOMITOKU output for combined text (hard requirement)
    t_combined = t_yomi

    result = {
        "text_pdf": t_pdf,
        "text_yomitoku": t_yomi,
        "text_combined": t_combined,
    }
    LOGGER.info("[OCR] combined length: %d", len(t_combined or ""))
    return result


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

# Accept half-width and full-width digits/symbols. We normalize before parsing, but
# keep a permissive regex as well to catch mixed strings.
YEN_AMOUNT_PATTERNS = [
    # Optional sign, optional Yen symbol, number (1,234 or 1234), optional 蜀・    re.compile(r"([\-竏綻?)\s*[ﾂ･・･]?\s*([0-9・・・兢{1,3}(?:[,・珪[0-9・・・兢{3})+|[0-9・・・兢+)\s*(?:蜀・?"),
]

# Translation table for full-width to half-width digits and punctuation used in amounts
_FW_TO_HW = str.maketrans({
    "・・: "0", "・・: "1", "・・: "2", "・・: "3", "・・: "4",
    "・・: "5", "・・: "6", "・・: "7", "・・: "8", "・・: "9",
    "・・: ",", "・・: ".", "・･": "ﾂ･", "・・: "-", "窶・: "-", "繝ｼ": "-",
})

def _normalize_amount_text(s: str) -> str:
    # Translate common full-width chars and collapse internal spaces
    s = s.translate(_FW_TO_HW)
    # Replace unusual thin/non-breaking spaces if present
    s = s.replace("\u2009", " ").replace("\u00A0", " ")
    # Some OCR introduces spaces between digits: "8 0 0" 竊・"800"
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    return s


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
    # Normalize then strip thousands separators
    s = _normalize_amount_text(s)
    s = s.replace(",", "")
    try:
        val = int(s)
        return -val if sign in ("-", "竏・) else val
    except ValueError:
        return 0


def _find_amount(text: str) -> Optional[int]:
    candidates: list[tuple[int, int]] = []
    for line in text.splitlines():
        # Keep original for keyword checks; normalize for number matching
        lowered = line.lower()
        norm_line = _normalize_amount_text(line)
        weight = 1
        if any(k in lowered for k in ("蜷郁ｨ・, "險・, "驥鷹｡・, "邊ｾ邂・, "隲区ｱる｡・, "遞手ｾｼ", "謾ｯ謇・)) or ("蜀・ in line or "ﾂ･" in line or "・･" in line):
            weight = 3
        for pat in YEN_AMOUNT_PATTERNS:
            for m in pat.finditer(norm_line):
                sign = m.group(1) or ""
                num = m.group(2)
                amount = _parse_int_amount(num, sign)
                if amount == 0:
                    continue
                # Heuristic: if the line has no currency/amount cue, ignore unusually long numbers (likely IDs)
                if weight < 3 and len(re.sub(r"\D", "", num)) > 6:
                    continue
                candidates.append((weight * abs(amount), amount))
    if not candidates:
        return None
    _, best_amount = max(candidates, key=lambda t: t[0])
    return best_amount


VENDOR_KEYWORDS = {
    "amazon": "豸郁怜刀雋ｻ",
    "繧｢繝槭だ繝ｳ": "豸郁怜刀雋ｻ",
    "讌ｽ螟ｩ": "豸郁怜刀雋ｻ",
    "繝､繝槭ヨ": "闕ｷ騾驕玖ｳ・,
    "菴仙ｷ・: "闕ｷ騾驕玖ｳ・,
    "繧・≧繝代ャ繧ｯ": "闕ｷ騾驕玖ｳ・,
    "驛ｵ萓ｿ": "騾壻ｿ｡雋ｻ",
    "蛻・焔": "騾壻ｿ｡雋ｻ",
    "繧ｿ繧ｯ繧ｷ繝ｼ": "譌・ｲｻ莠､騾夊ｲｻ",
    "uber": "譌・ｲｻ莠､騾夊ｲｻ",
    "jr": "譌・ｲｻ莠､騾夊ｲｻ",
    "髮ｻ霆・: "譌・ｲｻ莠､騾夊ｲｻ",
    "繧ｬ繧ｽ繝ｪ繝ｳ": "霆贋ｸ｡雋ｻ",
    "eneos": "霆贋ｸ｡雋ｻ",
    "蜃ｺ蜑埼､ｨ": "莨夊ｭｰ雋ｻ",
    "繧ｦ繝ｼ繝舌・繧､繝ｼ繝・: "莨夊ｭｰ雋ｻ",
    "繝槭け繝峨リ繝ｫ繝・: "莨夊ｭｰ雋ｻ",
    "繧ｹ繧ｿ繝ｼ繝舌ャ繧ｯ繧ｹ": "莠､髫幄ｲｻ",
}

PAYMENT_KEYWORDS = {
    "繧ｯ繝ｬ繧ｸ繝・ヨ": "譛ｪ謇暮≡",
    "visa": "譛ｪ謇暮≡",
    "mastercard": "譛ｪ謇暮≡",
    "jcb": "譛ｪ謇暮≡",
    "amex": "譛ｪ謇暮≡",
    "paypay": "譛ｪ謇暮≡",
    "line pay": "譛ｪ謇暮≡",
    "讌ｽ螟ｩ繝壹う": "譛ｪ謇暮≡",
    "隲区ｱよ嶌": "譛ｪ謇暮≡",
    "謖ｯ霎ｼ": "譎ｮ騾夐宣≡",
    "謖ｯ譖ｿ": "譎ｮ騾夐宣≡",
    "蜈･驥・: "譎ｮ騾夐宣≡",
    "蠑戊誠": "譎ｮ騾夐宣≡",
    "迴ｾ驥・: "迴ｾ驥・,
    "繝ｬ繧ｸ": "迴ｾ驥・,
    "atm": "譎ｮ騾夐宣≡",
    "驫陦・: "譎ｮ騾夐宣≡",
}


def _guess_accounts(text: str) -> Tuple[str, str]:
    t = text.lower()
    for kw, acc in VENDOR_KEYWORDS.items():
        if kw in t:
            debit = acc
            break
    else:
        debit = "髮題ｲｻ"

    for kw, acc in PAYMENT_KEYWORDS.items():
        if kw in t:
            credit = acc
            break
    else:
        credit = "譛ｪ謇暮≡"
    return debit, credit


def _extract_counterparty(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = lines[:25]
    best = ""
    for ln in head:
        if len(ln) < 3:
            continue
        if re.search(r"[A-Za-z繧｡-繝ｳ・ｧ-・晢ｾ橸ｾ滉ｸ-鮴･]{2,}", ln):
            if any(k in ln for k in ("鬆伜庶", "隲区ｱ・, "蜷郁ｨ・, "驥鷹｡・, "譏守ｴｰ", "蜀・ｨｳ")):
                continue
            best = ln
            break
    return best[:64]


def extract_journal_data(text: str) -> ParsedJournal:
    date = _find_date_smart(text)
    amount = _find_amount_smart(text)
    debit, credit = _guess_accounts(text)
    counterparty = _extract_counterparty_smart(text)
    summary_parts = [p for p in [counterparty or None, "雉ｼ蜈･"] if p]
    summary = " ".join(summary_parts) if summary_parts else "謾ｯ謇・
    return ParsedJournal(
        date=date,
        amount=amount,
        summary=summary,
        debit_account=debit,
        credit_account=credit,
        counterparty=counterparty,
    )


# Improved helpers: smarter date/amount/counterparty detection
def _find_date_smart(text: str) -> Optional[str]:
    try:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        labeled = [
            re.compile(r"(逋ｺ陦梧律|縺碑ｫ区ｱよ律|隲区ｱよ律|縺泌茜逕ｨ譌･|蛻ｩ逕ｨ譌･|雉ｼ蜈･譌･|縺願ｲｷ荳頑律|鬆伜庶譌･|蜿門ｼ墓律|豕ｨ譁・律|邏榊刀譌･|譌･莉・[:・咯?\s*(\d{4})[\./蟷ｴ・十-](\d{1,2})[\./譛茨ｼ十-](\d{1,2})譌･?"),
            re.compile(r"(逋ｺ陦梧律|縺碑ｫ区ｱよ律|隲区ｱよ律|縺泌茜逕ｨ譌･|蛻ｩ逕ｨ譌･|雉ｼ蜈･譌･|縺願ｲｷ荳頑律|鬆伜庶譌･|蜿門ｼ墓律|豕ｨ譁・律|邏榊刀譌･|譌･莉・[:・咯?\s*(\d{4})(\d{2})(\d{2})"),
        ]
        for ln in lines:
            for pat in labeled:
                m = pat.search(ln)
                if m and len(m.groups()) >= 4:
                    y, mm, dd = m.group(2), m.group(3), m.group(4)
                    d = _norm_date(y, mm, dd)
                    if d:
                        return d
        ignore = ("繧ｹ繧ｭ繝｣繝ｳ", "ScanSnap", "菴懈・", "逕滓・", "蜃ｺ蜉・, "蜊ｰ蛻ｷ", "菫晏ｭ・, "繧｢繝・・繝ｭ繝ｼ繝・, "download", "uploaded")
        generic = [
            re.compile(r"(20\d{2})[\-/\.](\d{1,2})[\-/\.](\d{1,2})"),
            re.compile(r"(\d{4})(\d{2})(\d{2})"),
        ]
        for ln in lines:
            low = ln.lower()
            if any(k.lower() in low for k in ignore):
                continue
            for pat in generic:
                m = pat.search(ln)
                if m and len(m.groups()) == 3:
                    y, mm, dd = m.groups()
                    d = _norm_date(y, mm, dd)
                    if d:
                        return d
    except Exception:
        pass
    return _find_date(text)


def _find_amount_smart(text: str) -> Optional[int]:
    try:
        candidates: list[tuple[int, int]] = []
        cues = ("蜷郁ｨ・, "蜷育ｮ・, "險・, "隲区ｱ・, "驥鷹｡・, "遞手ｾｼ", "遞取栢", "縺頑髪謇・, "縺碑ｫ区ｱ・, "縺皮ｲｾ邂・, "縺願ｲｷ荳・, "縺碑ｳｼ蜈･")
        pat = re.compile(r"([\-竏綻?)\s*[ﾂ･\u00A5]?\s*([0-9]{1,3}(?:[,\s][0-9]{3})+|[0-9]+)\s*(?:蜀・?")
        for line in text.splitlines():
            norm = _normalize_amount_text(line)
            weight = 1
            if any(k in line for k in cues) or ("蜀・ in line) or ("ﾂ･" in line):
                weight = 5
            for m in pat.finditer(norm):
                sign = m.group(1) or ""
                num = m.group(2)
                amount = _parse_int_amount(num, sign)
                if amount == 0:
                    continue
                if weight < 5 and len(re.sub(r"\D", "", num)) > 6:
                    continue
                candidates.append((weight * abs(amount), amount))
        if candidates:
            return max(candidates, key=lambda t: t[0])[1]
    except Exception:
        pass
    return _find_amount(text)


def _extract_counterparty_smart(text: str) -> str:
    try:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines[:25]:
            if len(ln) < 3:
                continue
            if ln.startswith("%PDF-") or ln.upper().startswith("PDF-"):
                continue
            if any(bad in ln for bad in ("YOMITOKU", "Adobe", "Image", "ScanSnap", "FUJITSU")):
                continue
            if re.search(r"[A-Za-z繧｡-繝ｳ荳-鮴･縲・・]{2,}", ln):
                if any(k in ln for k in ("鬆伜庶", "隲区ｱ・, "蜷郁ｨ・, "驥鷹｡・, "譏守ｴｰ", "蜈･驥・, "蜃ｺ驥・)):
                    continue
                return ln[:64]
    except Exception:
        pass
    return _extract_counterparty(text)
