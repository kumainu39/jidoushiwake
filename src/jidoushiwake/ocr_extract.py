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
    """Extract text with graceful fallbacks.

    Order of preference:
    1) YOMITOKU (CLI or package)
    2) PaddleOCR rasterized from PDF pages
    3) Embedded PDF text (pdfminer/PyPDF2)

    Returns a dict with keys: text_pdf, text_yomitoku, text_combined.
    """
    LOGGER.info("[OCR] begin extract_text_both: %s", pdf_path)

    t_pdf = ""
    t_yomi = ""
    t_paddle = ""

    # 1) Try YOMITOKU first, but do not crash if empty
    try:
        t_yomi = extract_text_with_yomitoku(pdf_path, ensure=False)
    except Exception as e:
        LOGGER.warning("[OCR] YOMITOKU failed: %s", e)
        t_yomi = ""
    LOGGER.info("[OCR] YOMITOKU extracted: %d chars", len(t_yomi or ""))

    # 2) If YOMITOKU empty, try PaddleOCR
    if not (t_yomi and t_yomi.strip()):
        try:
            t_paddle = _extract_text_with_paddle(pdf_path)
        except Exception as e:
            LOGGER.warning("[OCR] PaddleOCR fallback failed: %s", e)
            t_paddle = ""
        LOGGER.info("[OCR] PaddleOCR extracted: %d chars", len(t_paddle or ""))

    # 3) As a last resort, try embedded PDF text
    if not (t_yomi and t_yomi.strip()) and not (t_paddle and t_paddle.strip()):
        try:
            t_pdf = extract_text_from_pdf(pdf_path)
        except Exception as e:
            LOGGER.warning("[OCR] PDF text fallback failed: %s", e)
            t_pdf = ""
        LOGGER.info("[OCR] PDF embedded extracted: %d chars", len(t_pdf or ""))

    # Choose best available for combined
    t_combined = (t_yomi or t_paddle or t_pdf or "")

    result = {
        "text_pdf": t_pdf,
        "text_yomitoku": t_yomi,
        "text_combined": t_combined,
    }
    # Optionally include Paddle text for future use
    try:
        result["text_paddle"] = t_paddle
    except Exception:
        pass

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


# ---- Japanese era (和暦) support ----
# Full-width to half-width translation for digits and common separators used in dates
_DATE_FW_TO_HW = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "／": "/", "．": ".", "－": "-", "ー": "-", "―": "-",
    "年": "年", "月": "月", "日": "日", "Ｒ": "R", "Ｈ": "H", "Ｓ": "S", "Ｔ": "T", "Ｍ": "M",
})

# Era to Gregorian base year mapping (first year of era)
_ERA_BASE = {
    "令和": 2019,
    "平成": 1989,
    "昭和": 1926,
    "大正": 1912,
    "明治": 1868,
    "R": 2019,
    "H": 1989,
    "S": 1926,
    "T": 1912,
    "M": 1868,
}

def _normalize_date_text(s: str) -> str:
    try:
        s = s.translate(_DATE_FW_TO_HW)
        s = re.sub(r"\s*([/\.\-年月日])\s*", r"\1", s)
    except Exception:
        pass
    return s

def _wareki_to_ymd(era: str, year_text: str, month_text: str, day_text: str) -> Optional[str]:
    try:
        base = _ERA_BASE.get(era)
        if base is None:
            return None
        y = 1 if year_text in ("元", "元年") else int(year_text)
        g = base + y - 1
        return _norm_date(str(g), month_text, day_text)
    except Exception:
        return None

def _find_date_wareki(text: str) -> Optional[str]:
    """Find a date written in Japanese era (和暦) and convert to YYYY/MM/DD."""
    try:
        lines = [_normalize_date_text(ln.strip()) for ln in text.splitlines() if ln.strip()]
        pat_kanji = re.compile(r"(令和|平成|昭和|大正|明治)(元|\d{1,2})年(\d{1,2})月(\d{1,2})日?")
        pat_kanji_sep = re.compile(r"(令和|平成|昭和|大正|明治)(元|\d{1,2})[/.\-](\d{1,2})[/.\-](\d{1,2})")
        pat_initial = re.compile(r"([RrHhSsTtMm])\.?\s*(\d{1,2})[/.\-年](\d{1,2})[/.\-月](\d{1,2})日?")
        for ln in lines:
            m = pat_kanji.search(ln)
            if m:
                d = _wareki_to_ymd(m.group(1), m.group(2), m.group(3), m.group(4))
                if d:
                    return d
            m = pat_kanji_sep.search(ln)
            if m:
                d = _wareki_to_ymd(m.group(1), m.group(2), m.group(3), m.group(4))
                if d:
                    return d
            m = pat_initial.search(ln)
            if m:
                era = m.group(1).upper()
                d = _wareki_to_ymd(era, m.group(2), m.group(3), m.group(4))
                if d:
                    return d
    except Exception:
        pass
    return None

def _find_date(text: str) -> Optional[str]:
    # Try era-based date first
    d_era = _find_date_wareki(text)
    if d_era:
        return d_era
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
        credit = "現金"
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
        # If the document uses Japanese era dates, convert first
        d_era_all = _find_date_wareki(text)
        if d_era_all:
            return d_era_all
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
        # Generic Japanese date without label: YYYY年M月D日（曜日等の括弧は無視）
        try:
            pat_ja = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
            for ln in lines:
                m = pat_ja.search(ln)
                if m:
                    d = _norm_date(m.group(1), m.group(2), m.group(3))
                    if d:
                        return d
        except Exception:
            pass
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
    amount = _find_amount_best(text)
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


def _find_amount_best(text: str) -> Optional[int]:
    """Improved chooser that prefers bottom-most cued totals and logs selection.

    Falls back to existing smart and simple detectors if heuristics find nothing.
    """
    try:
        import re as _re

        # Label-aware extraction: remember last seen label and attach to amounts
        total_labels = ("合計", "お買上げ合計", "お会計", "総計", "ご請求金額", "請求金額")
        subtotal_labels = ("小計",)
        tax_labels = ("消費税", "内消費税", "外税", "内税", "税")
        recv_labels = ("お預り", "お預かり", "お支払い", "受領", "現金")
        change_labels = ("お釣", "釣銭", "おつり")

        excl = ("口座", "登録番号", "POS", "TEL", "電話", "FAX", "〒", "JAN", "コード", "Code", "No", "No.", "番号", "伝票", "注", "会員", "顧客")

        def label_of(line: str) -> str | None:
            l = line.replace(" ", "")
            if any(k in l for k in total_labels) or _re.search(r"合.{0,2}計", l):
                return "total"
            if any(k in l for k in subtotal_labels):
                return "subtotal"
            if any(k in l for k in tax_labels):
                return "tax"
            if any(k in l for k in recv_labels):
                return "recv"
            if any(k in l for k in change_labels):
                return "change"
            return None

        pat = _re.compile(r"([\-\u2212]?)\s*[¥\u00A5]?\s*([0-9]{1,3}(?:[, \u00A0][0-9]{3})+|[0-9]+)\s*(?:円?)")
        lines = text.splitlines()
        last_lab: str | None = None
        labeled: list[dict] = []
        for idx, line in enumerate(lines):
            if any(k in line for k in excl):
                # Skip obvious non-amount contexts entirely
                continue
            # Exclude likely phone numbers (e.g., 06-6774-4180)
            try:
                if _re.search(r"\b0\d{1,4}-\d{2,4}-\d{3,4}\b", line):
                    continue
            except Exception:
                pass
            lab = label_of(line) or last_lab
            if label_of(line):
                last_lab = label_of(line)
            norm = _normalize_amount_text(line)
            for m in pat.finditer(norm):
                sign = m.group(1) or ""
                num = m.group(2)
                amt = _parse_int_amount(num, sign)
                if amt == 0:
                    continue
                # Ignore quantities like "3個"等 when no currency symbol is present
                if ("円" not in line and "¥" not in line and "￥" not in line) and any(u in line for u in ("個", "点", "枚", "本", "台")):
                    continue
                labeled.append({"idx": idx, "amount": amt, "label": lab, "line": line})

        # If document mentions 合計金額 but no explicit total amount, sum item nets
        try:
            if any("合計金額" in l for l in lines) and not any(r.get("label") == "total" for r in labeled):
                nets: list[int] = []
                for i, ln in enumerate(lines):
                    nn = _normalize_amount_text(ln)
                    # Match numbers ending with 円 and not closed by ')'
                    m_net = _re.search(r"([0-9]{1,3}(?:[, \u00A0][0-9]{3})+|[0-9]+)\s*円(?!\))", nn)
                    if m_net:
                        try:
                            nets.append(int(_re.sub(r"\D", "", m_net.group(1))))
                        except Exception:
                            pass
                if len(nets) >= 2:
                    total_sum = sum(nets)
                    try:
                        LOGGER.info("[OCR] composed total from nets due to 合計金額: %s", total_sum)
                    except Exception:
                        pass
                    return total_sum
        except Exception:
            pass

        # Prefer explicit total → (subtotal+tax) → recent subtotal
        totals = [r for r in labeled if r.get("label") == "total"]
        if totals:
            chosen = max(totals, key=lambda r: r["idx"])  # bottom-most total
            try:
                LOGGER.info("[OCR] amount chosen (label=total): %s from line: %s", chosen["amount"], chosen["line"][:120])
            except Exception:
                pass
            return int(chosen["amount"]) if chosen else None

        # Try to combine subtotal + tax when they appear
        subs = [r for r in labeled if r.get("label") == "subtotal"]
        taxes = [r for r in labeled if r.get("label") == "tax"]
        if subs and taxes:
            # Choose nearest pair (tax after subtotal within 5 lines, else last pair)
            best = None
            best_dist = 1e9
            for s in subs:
                for t in taxes:
                    if t["idx"] >= s["idx"] and (t["idx"] - s["idx"]) <= 5:
                        d = t["idx"] - s["idx"]
                        if d < best_dist:
                            best = (s, t)
                            best_dist = d
            if not best:
                best = (subs[-1], taxes[-1])
            s, t = best
            gross = int(s["amount"]) + int(t["amount"])
            try:
                LOGGER.info("[OCR] amount composed from subtotal+tax: %s + %s = %s", s["amount"], t["amount"], gross)
            except Exception:
                pass
            return gross

        if subs:
            # If we only see subtotal (and items marked as 込), subtotal is likely gross
            chosen = subs[-1]
            try:
                LOGGER.info("[OCR] amount chosen (label=subtotal): %s from line: %s", chosen["amount"], chosen["line"][:120])
            except Exception:
                pass
            return int(chosen["amount"]) if chosen else None

        # As a last resort, pick the last yen-amount (line contains 円/¥) not labeled as recv/change/tax
        fallback = [r for r in labeled if r.get("label") not in ("recv", "change", "tax") and ("円" in r.get("line","") or "¥" in r.get("line","") or "￥" in r.get("line",""))]
        if fallback:
            chosen = fallback[-1]
            try:
                LOGGER.info("[OCR] amount chosen (fallback): %s from line: %s", chosen["amount"], chosen["line"][:120])
            except Exception:
                pass
            return int(chosen["amount"]) if chosen else None
    except Exception:
        pass
    # Fallback to existing detectors
    return _find_amount_smart(text) or _find_amount(text)
