import argparse
import logging
from logging.handlers import RotatingFileHandler
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Ensure local src package is importable when executed directly
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jidoushiwake.yayoi_exporter import (
    AccountSide,
    JournalEntry,
    JournalExporter,
)


LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "scansnap_ocr.log"
OUTPUT_DIR = BASE_DIR / "output"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(LOG_FILE, maxBytes=512_000, backupCount=3, encoding="utf-8")
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from an OCR'ed PDF using pdfminer if available, else PyPDF2.

    Parameters
    ----------
    pdf_path: Path to the OCR'ed PDF file.
    """

    logging.info("Extracting text from PDF: %s", pdf_path)

    # Try pdfminer.six first (more accurate for OCR text layers)
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore

        text = pdfminer_extract_text(str(pdf_path))
        if text and text.strip():
            logging.info("Text extracted via pdfminer (%d chars)", len(text))
            return text
    except Exception as e:  # noqa: BLE001 - best-effort optional dependency
        logging.warning("pdfminer extract failed: %s", e)

    # Fallback to PyPDF2
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
        logging.info("Text extracted via PyPDF2 (%d chars)", len(text))
        return text
    except Exception as e:  # noqa: BLE001
        logging.error("PyPDF2 extract failed: %s", e)
        return ""


@dataclass
class ParsedJournal:
    date: Optional[str]
    amount: Optional[int]
    summary: str
    debit_account: str
    credit_account: str
    counterparty: str = ""


DATE_PATTERNS = [
    re.compile(r"(20\d{2})[\-/\.](\d{1,2})[\-/\.](\d{1,2})"),  # 2025-10-22
    re.compile(r"(\d{4})(\d{2})(\d{2})"),  # 20251022
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
            # disambiguate compact pattern: groups are y,m,d
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
    # Prefer lines containing typical amount labels
    candidates: list[Tuple[int, int]] = []  # (abs_amount, signed_amount)
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
    # Pick the highest-weighted amount
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
    # credit side guess
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
    # Debit (expense) account guess from vendor keywords
    for kw, acc in VENDOR_KEYWORDS.items():
        if kw in t:
            debit = acc
            break
    else:
        debit = "雑費"

    # Credit (payment source) account guess
    for kw, acc in PAYMENT_KEYWORDS.items():
        if kw in t:
            credit = acc
            break
    else:
        credit = "未払金"

    return debit, credit


def _extract_counterparty(text: str) -> str:
    # Heuristic: pick a candidate from top lines containing katakana/latin words
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = lines[:25]
    best = ""
    for ln in head:
        if len(ln) < 3:
            continue
        if re.search(r"[A-Za-zァ-ンｧ-ﾝﾞﾟ一-龥]{2,}", ln):
            # exclude lines that look like headings
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

    # Build summary
    summary_parts = [p for p in [counterparty or None, "購入"] if p]
    summary = " ".join(summary_parts) if summary_parts else "支払"

    logging.info(
        "Parsed fields - date=%s amount=%s debit=%s credit=%s counterparty=%s",
        date,
        amount,
        debit,
        credit,
        counterparty,
    )

    return ParsedJournal(
        date=date,
        amount=amount,
        summary=summary,
        debit_account=debit,
        credit_account=credit,
        counterparty=counterparty,
    )


def build_yayoi_entry(parsed: ParsedJournal, src_pdf: Path) -> JournalEntry:
    # Fallbacks
    date = parsed.date or datetime.now().strftime("%Y/%m/%d")
    amount = parsed.amount or 0

    debit = AccountSide(account=parsed.debit_account, amount=abs(amount))
    credit = AccountSide(account=parsed.credit_account, amount=abs(amount))

    entry = JournalEntry(
        identifier_flag="****",  # Yayoi import identifier (kept generic)
        voucher_number="1",
        transaction_date=date,
        debit=debit,
        credit=credit,
        summary=parsed.summary[:64],
        reference_number=src_pdf.stem[:10],
        memo=str(src_pdf.name)[:180],
        type_code="0",
        origin_code="",
        tag1="",
        tag2="",
        adjustment_flag=False,
        closing_classification="",
    )
    return entry


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract OCR text from PDF and export Yayoi CSV")
    parser.add_argument("pdf_path", help="Path to OCR'ed PDF file from ScanSnap Home")
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Output CSV encoding (e.g., utf-8, utf-8-sig)",
    )
    args = parser.parse_args(argv)

    setup_logging()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logging.error("PDF not found: %s", pdf_path)
        return 2

    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            logging.error("No text extracted from PDF (ensure OCR text layer exists)")
            return 3

        parsed = extract_journal_data(text)
        entry = build_yayoi_entry(parsed, pdf_path)

        # Output path: output/YYYYMMDD_yayoi.csv (use parsed date or today)
        date_for_name = (parsed.date or datetime.now().strftime("%Y/%m/%d")).replace("/", "")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{date_for_name}_yayoi.csv"

        exporter = JournalExporter(entries=[entry])
        exporter.save(out_path, encoding=args.encoding, bom=args.encoding.lower() == "utf-8-sig")
        logging.info("CSV saved: %s", out_path)
        return 0
    except Exception as e:  # noqa: BLE001
        logging.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

