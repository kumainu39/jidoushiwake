from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import os


LOGGER = logging.getLogger(__name__)

# Externalized YOMITOKU integration
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


def extract_text_both(pdf_path: Path) -> dict:
    """Extract text via embedded text layer and YOMITOKU (PaddleOCR optional).

    Returns a dict with keys: text_pdf, text_yomitoku, text_combined.

    text_combined always uses YOMITOKU output.
    """
    LOGGER.info("[OCR] begin extract_text_both: %s", pdf_path)
    t_pdf = extract_text_from_pdf(pdf_path)
    LOGGER.info("[OCR] pdfminer/PyPDF2 extracted: %d chars", len(t_pdf or ""))
    t_yomi = extract_text_with_yomitoku(pdf_path, ensure=True)
    LOGGER.info("[OCR] YOMITOKU extracted: %d chars", len(t_yomi or ""))
    if not (t_yomi and t_yomi.strip()):
        raise RuntimeError(
            "YOMITOKU OCR is required but not available or produced no text. "
            "Install and ensure 'yomitoku' CLI (or Python package) works. "
            "You can set YOMITOKU_EXE to the CLI path."
        )

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
    # Optional sign, optional Yen symbol, number (1,234 or 1234), optional 円
    re.compile(r"([\-−]?)\s*[¥￥]?\s*([0-9０-９]{1,3}(?:[,，][0-9０-９]{3})+|[0-9０-９]+)\s*(?:円)?"),
]

# Translation table for full-width to half-width digits and punctuation used in amounts
_FW_TO_HW = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "，": ",", "．": ".", "￥": "¥", "－": "-", "―": "-", "ー": "-",
})


def _normalize_amount_text(s: str) -> str:
    # Translate common full-width chars and collapse internal spaces
    s = s.translate(_FW_TO_HW)
    # Replace unusual thin/non-breaking spaces if present
    s = s.replace("\u2009", " ").replace("\u00A0", " ")
    # Some OCR introduces spaces between digits: "8 0 0" → "800"
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
        return -val if sign in ("-", "−") else val
    except ValueError:
        return 0


def _find_amount(text: str) -> Optional[int]:
    candidates: list[tuple[int, int]] = []
    for line in text.splitlines():
        # Keep original for keyword checks; normalize for number matching
        lowered = line.lower()
        norm_line = _normalize_amount_text(line)
        weight = 1
        if any(k in lowered for k in ("合計", "計", "金額", "精算", "請求額", "税込", "支払")) or ("円" in line or "¥" in line or "￥" in line):
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


# Improved helpers: smarter date/amount/counterparty detection
def _find_date_smart(text: str) -> Optional[str]:
    try:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        labeled = [
            re.compile(r"(発行日|ご請求日|請求日|ご利用日|利用日|購入日|お買上日|領収日|取引日|注文日|納品日|日付)[:：]?\s*(\d{4})[\./年／\-](\d{1,2})[\./月／\-](\d{1,2})日?"),
            re.compile(r"(発行日|ご請求日|請求日|ご利用日|利用日|購入日|お買上日|領収日|取引日|注文日|納品日|日付)[:：]?\s*(\d{4})(\d{2})(\d{2})"),
        ]
        for ln in lines:
            for pat in labeled:
                m = pat.search(ln)
                if m and len(m.groups()) >= 4:
                    y, mm, dd = m.group(2), m.group(3), m.group(4)
                    d = _norm_date(y, mm, dd)
                    if d:
                        return d
        ignore = ("スキャン", "ScanSnap", "作成", "生成", "出力", "印刷", "保存", "アップロード", "download", "uploaded")
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
        cues = ("合計", "計", "請求", "請求額", "金額", "税込", "税抜", "お支払", "ご請求", "ご精算", "お買上", "ご購入")
        pat = re.compile(r"([\-−]?)\s*[¥\u00A5]?\s*([0-9]{1,3}(?:[,\s][0-9]{3})+|[0-9]+)\s*(?:円)?")
        for line in text.splitlines():
            norm = _normalize_amount_text(line)
            weight = 1
            if any(k in line for k in cues) or ("円" in line) or ("¥" in line) or ("￥" in line):
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
            if re.search(r"[A-Za-zァ-ンｧ-ﾝﾞﾟ一-龥]{2,}", ln):
                if any(k in ln for k in ("領収", "請求", "合計", "金額", "明細", "入金", "出金")):
                    continue
                return ln[:64]
    except Exception:
        pass
    return _extract_counterparty(text)


def extract_journal_data(text: str) -> ParsedJournal:
    date = _find_date_smart(text)
    amount = _find_amount_smart(text)
    debit, credit = _guess_accounts(text)
    counterparty = _extract_counterparty_smart(text)
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

