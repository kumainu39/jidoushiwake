from __future__ import annotations

from datetime import datetime
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import os

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_session, engine, Base
from .models import (
    AutoResult,
    AutoResultLine,
    Company,
    Correction,
    CorrectionActionEnum,
    Document,
    DocumentStatusEnum,
    KeywordMapping,
    GlobalRule,
    CompanySetting,
    CompanyLLMSetting,
    LLMLog,
    AccountSetting,
    TaxCategory,
    AccountMaster,
)
from .ocr_extract import extract_text_from_pdf, extract_journal_data, extract_text_both
from .yayoi_exporter import AccountSide, JournalEntry, JournalExporter
from .llm_client import refine_extraction, refine_extraction_with_yomi
from .llm_client import parse_nl_rule


def init_db() -> None:
    """Initialize database and run lightweight, inline migrations.

    Note: SQLAlchemy's ``create_all`` does not add columns to existing tables.
    For SQLite users upgrading from an older version, we proactively add the
    ``archive_base_dir`` column to ``company_settings`` if it is missing.
    """
    Base.metadata.create_all(engine)

    # Minimal, idempotent migration for SQLite: add missing columns
    try:
        from sqlalchemy import text

        with engine.begin() as conn:
            # Check existing columns
            cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info('company_settings')").all()]
            if "archive_base_dir" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE company_settings ADD COLUMN archive_base_dir TEXT"
                )
            # AutoResult new columns
            a_cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info('auto_results')").all()]
            if "debit_subaccount" not in a_cols:
                conn.exec_driver_sql("ALTER TABLE auto_results ADD COLUMN debit_subaccount VARCHAR(64)")
            if "credit_subaccount" not in a_cols:
                conn.exec_driver_sql("ALTER TABLE auto_results ADD COLUMN credit_subaccount VARCHAR(64)")
            if "invoice_status" not in a_cols:
                conn.exec_driver_sql("ALTER TABLE auto_results ADD COLUMN invoice_status VARCHAR(16)")
            # CompanyLLMSetting new columns
            cl_cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info('company_llm_settings')").all()]
            if "use_colab" not in cl_cols:
                conn.exec_driver_sql("ALTER TABLE company_llm_settings ADD COLUMN use_colab INTEGER DEFAULT 0")
            if "remote_base_url" not in cl_cols:
                conn.exec_driver_sql("ALTER TABLE company_llm_settings ADD COLUMN remote_base_url TEXT")
            # tax category stored as short code in invoice_status for now; also ensure tax_categories table exists
            conn.exec_driver_sql(
                "CREATE TABLE IF NOT EXISTS tax_categories (\n"
                "id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                "code VARCHAR(32) UNIQUE,\n"
                "name VARCHAR(128),\n"
                "abbrev VARCHAR(64),\n"
                "active INTEGER DEFAULT 1\n"
                ")"
            )
    except Exception:
        # Never block app startup due to migration helper; the API will surface
        # errors if the column truly cannot be added (e.g., permissions).
        pass

    # Seed tax categories if empty
    with get_session() as s:
        try:
            cnt = s.query(TaxCategory).count()
        except Exception:
            cnt = 0
        if cnt == 0:
            seeds = [
                ("TAISHOGAI", "対象外", "対象外"),
                ("KZ_URI_10", "課税売上10%", "課税売上10%"),
                ("KZ_URI_8K", "課税売上8%（軽）", "課税売上8%(軽)"),
                ("KZ_SHI_10", "課税仕入10%", "課対仕入10%"),
                ("KZ_SHI_8K", "課税仕入8%（軽）", "課対仕入8%(軽)"),
                ("HIZEI_URI", "非課税売上", "非課売上"),
                ("HIZEI_SHI", "非課税仕入", "非課仕入"),
                ("YUSHUTS_URI", "輸出売上", "輸出売上"),
                ("YUSHUTS_SHI", "輸出仕入", "輸出仕入"),
            ]
            for code, name, abbr in seeds:
                s.add(TaxCategory(code=code, name=name, abbrev=abbr, active=1))
            s.flush()


def ensure_company(session: Session, name: str) -> Company:
    company = session.scalar(select(Company).where(Company.name == name))
    if company:
        return company
    company = Company(name=name)
    session.add(company)
    session.flush()
    return company


def list_companies(session: Session) -> list[Company]:
    return list(session.scalars(select(Company).order_by(Company.name)))


def import_pdf(session: Session, company_id: int, pdf_path: Path) -> Document:
    texts = extract_text_both(pdf_path)
    text_combined = texts.get("text_combined") or ""
    parsed = extract_journal_data(text_combined)

    # Optionally refine with LLM using company overrides (GPU-only in runtime)
    try:
        # Load company override if any
        c_llm = session.scalar(select(CompanyLLMSetting).where(CompanyLLMSetting.company_id == company_id))
        llm_ref = None
        if c_llm and c_llm.use_override:
            from .llm_client import LLMConfig, temporary_config

            cfg = LLMConfig(
                provider=c_llm.provider,
                model_path=c_llm.model_path,
                device="gpu",
                n_gpu_layers=c_llm.n_gpu_layers,
                n_threads=c_llm.n_threads,
                lora_path=c_llm.lora_path,
                prompt_template=c_llm.prompt_template,
            )
            with temporary_config(cfg):
                llm_ref = _run_llm_refine_paginated(session, company_id, texts)
        else:
            llm_ref = _run_llm_refine_paginated(session, company_id, texts)
        if llm_ref:
            parsed.date = llm_ref.get("date") or parsed.date
            # Amount: prefer LLM inference when available (heuristics often miss true totals)
            if llm_ref.get("amount") is not None:
                try:
                    parsed.amount = int(llm_ref.get("amount"))
                except Exception:
                    parsed.amount = llm_ref.get("amount")
            parsed.summary = llm_ref.get("summary") or parsed.summary
            parsed.debit_account = llm_ref.get("debit_account") or parsed.debit_account
            parsed.credit_account = llm_ref.get("credit_account") or parsed.credit_account
            # Log LLM output
            try:
                import json as _json

                log = LLMLog(
                    company_id=company_id,
                    document_id=0,  # temporary until doc persisted below
                    model_id=(c_llm.model_path if c_llm and c_llm.use_override else None),
                    device="gpu",
                    prompt_excerpt=(texts.get("text_pdf") or "")[:500],
                    response_json=_json.dumps(llm_ref, ensure_ascii=False),
                    confidence=float(llm_ref.get("confidence")) if llm_ref.get("confidence") is not None else None,
                )
                session.add(log)
                session.flush()
            except Exception:
                pass
    except Exception:
        pass

    doc = Document(company_id=company_id, file_path=str(pdf_path), ocr_text=text_combined)
    session.add(doc)
    session.flush()
    # Update LLM log with actual document_id
    try:
        for l in session.scalars(select(LLMLog).where(LLMLog.company_id == company_id, LLMLog.document_id == 0)):
            l.document_id = doc.id
            session.add(l)
    except Exception:
        pass

    # Apply global rules (initial defaults) then per-company mappings to refine accounts
    debit = parsed.debit_account
    credit = parsed.credit_account
    applied_score = 0

    # First, apply enabled GlobalRule by highest priority if accounts are not already set
    try:
        text_lower = (text_combined or "").lower()
        if not (debit and credit):
            for gr in session.scalars(
                select(GlobalRule).where(GlobalRule.enabled == 1).order_by(GlobalRule.priority.desc(), GlobalRule.id.asc())
            ):
                if gr.keyword and (gr.keyword.lower() in text_lower):
                    debit = debit or gr.debit_account
                    credit = credit or gr.credit_account
                    # Only apply the first matching highest-priority rule as initial default
                    break
    except Exception:
        pass

    # Then, apply per-company keyword mappings (can override if explicitly set)
    for km in session.scalars(select(KeywordMapping).where(KeywordMapping.company_id == company_id)):
        if km.keyword and (km.keyword.lower() in text_lower):
            if km.debit_account:
                debit = km.debit_account
            if km.credit_account:
                credit = km.credit_account
            applied_score += km.weight

    # Detect qualified invoice registration number (T + 13 digits) and simple tax hints
    import re as _re
    inv_status = None
    if _re.search(r"T\d{13}", text_combined):
        inv_status = "適格"
    else:
        # If clearly an invoice/receipt but no T number, mark as 非適格（暫定）
        if _re.search(r"(領収書|請求書|領収|請求)", text_combined):
            inv_status = "非適格"
    # Tax rate hints (lightweight)
    t_lower = (text_combined or "").lower()
    if any(k in t_lower for k in ("非課税", "不課税", "対象外")):
        inv_status = "非課税"
    elif any(k in t_lower for k in ("軽減", "8%", "８％", "8％")):
        inv_status = "軽減8%"
    elif any(k in t_lower for k in ("10%", "１０％", "10％")):
        inv_status = inv_status or "課税10%"

    auto = AutoResult(
        document_id=doc.id,
        date=parsed.date,
        amount=parsed.amount or 0,
        summary=parsed.summary,
        debit_account=debit,
        credit_account=credit,
        counterparty=parsed.counterparty,
        score=applied_score,
        invoice_status=inv_status,
    )
    session.add(auto)
    return doc


def _norm_date_str(s: str | None) -> str | None:
    """Normalize various date strings to 'YYYY/MM/DD'. Returns None on failure.

    Accepts common Japanese formats like 'YYYY-MM-DD', 'YYYY/MM/DD',
    'YYYY年M月D日', 'YY/MM/DD', and compacts without leading zeros.
    """
    if not s:
        return None
    t = (s or "").strip()
    if not t:
        return None
    import re as _re
    try:
        # 年月日形式
        m = _re.match(r"^(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})日?$", t)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{y:04d}/{mo:02d}/{d:02d}"
        # 2桁年 -> 2000年代と仮定
        m = _re.match(r"^(\d{2})[年/\-](\d{1,2})[月/\-](\d{1,2})日?$", t)
        if m:
            y, mo, d = 2000 + int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{y:04d}/{mo:02d}/{d:02d}"
        # 8桁 or 6桁(YYMMDD)
        m = _re.match(r"^(\d{8})$", t)
        if m:
            y = int(t[0:4]); mo = int(t[4:6]); d = int(t[6:8])
            return f"{y:04d}/{mo:02d}/{d:02d}"
        m = _re.match(r"^(\d{6})$", t)
        if m:
            y = 2000 + int(t[0:2]); mo = int(t[2:4]); d = int(t[4:6])
            return f"{y:04d}/{mo:02d}/{d:02d}"
        # fallback: split by non-digits
        parts = [p for p in _re.split(r"\D+", t) if p]
        if len(parts) >= 3:
            y = int(parts[0]); mo = int(parts[1]); d = int(parts[2])
            if y < 100:
                y += 2000
            return f"{y:04d}/{mo:02d}/{d:02d}"
    except Exception:
        return None
    return None


def _int_from_amount_like(v: object) -> int | None:
    """Convert various amount-like values to int. Returns None if not parseable."""
    if v is None:
        return None
    try:
        if isinstance(v, (int,)):
            return int(v)
        s = str(v).strip()
        if not s:
            return None
        # remove currency and commas
        import re as _re
        s = _re.sub(r"[円,\s\\]", "", s)
        # parentheses as negative
        neg = s.startswith("(") and s.endswith(")")
        s = s.strip("()")
        if s in ("-", ""):
            return None
        x = int(float(s))
        return -x if neg else x
    except Exception:
        return None


def import_transactions_table(
    session: Session,
    company_id: int,
    rows: list[dict[str, object]],
    source_path: Path,
) -> int:
    """Import bank/card transaction rows as Documents + AutoResults.

    This creates one Document per row, stores a synthesized text in `ocr_text`
    for downstream rule matching, and fills AutoResult fields from columns.
    """
    # Preload rules/mappings once
    from sqlalchemy import select as _select
    gr_list: list[GlobalRule] = []
    try:
        for gr in session.scalars(
            _select(GlobalRule).where(GlobalRule.enabled == 1).order_by(GlobalRule.priority.desc(), GlobalRule.id.asc())
        ):
            gr_list.append(gr)
    except Exception:
        gr_list = []
    km_list: list[KeywordMapping] = list(session.scalars(_select(KeywordMapping).where(KeywordMapping.company_id == company_id)))

    def _get_first(row: dict[str, object], keys: list[str]) -> str | None:
        for k in keys:
            for kk in (k, k.lower(), k.upper()):
                if kk in row and row[kk] is not None:
                    s = str(row[kk]).strip()
                    if s:
                        return s
        return None

    imported = 0
    for idx, row in enumerate(rows):
        # Normalize keys: keep original and lower
        norm_row: dict[str, object] = {}
        for k, v in list(row.items()):
            norm_row[k] = v
            if isinstance(k, str):
                lk = k.strip().lower()
                if lk and lk not in norm_row:
                    norm_row[lk] = v

        date_raw = _get_first(
            norm_row,
            [
                "日付",
                "取引日",
                "年月日",
                "date",
                "transaction date",
                "value date",
                "取引年月日",
            ],
        )
        date = _norm_date_str(date_raw) if date_raw else None

        # Amount: prefer single '金額/amount'; else compute deposit - withdrawal
        amount = None
        amount = _int_from_amount_like(
            _get_first(norm_row, ["金額", "amount", "ご利用金額", "利用金額", "出金額", "支出", "withdrawal"])
        )
        if amount is None:
            dep = _int_from_amount_like(_get_first(norm_row, ["入金額", "収入", "deposit"])) or 0
            wdr = _int_from_amount_like(_get_first(norm_row, ["出金額", "支出", "withdrawal"])) or 0
            amount = dep - wdr if (dep or wdr) else None

        summary = _get_first(
            norm_row,
            ["摘要", "内容", "メモ", "備考", "詳細", "説明", "description", "明細", "用途", "品目", "subject"],
        ) or ""
        counterparty = _get_first(
            norm_row, ["相手先", "支払先", "振込先", "取引先", "payee", "相手方", "店名", "merchant", "名称"]
        ) or ""
        debit = _get_first(norm_row, ["借方科目", "借方勘定科目", "debit", "debit_account"]) or ""
        credit = _get_first(norm_row, ["貸方科目", "貸方勘定科目", "credit", "credit_account"]) or ""

        # Build text blob for rule matching
        try:
            values_str = " ".join([str(v) for v in row.values() if v is not None])
        except Exception:
            values_str = summary or counterparty
        text_combined = f"{summary} {counterparty} {values_str}".strip()
        text_lower = (text_combined or "").lower()

        # Apply rules if accounts missing
        applied_score = 0
        if not (debit and credit):
            for gr in gr_list:
                try:
                    if gr.keyword and (gr.keyword.lower() in text_lower):
                        debit = debit or gr.debit_account
                        credit = credit or gr.credit_account
                        break
                except Exception:
                    continue
        for km in km_list:
            try:
                if km.keyword and (km.keyword.lower() in text_lower):
                    if km.debit_account:
                        debit = km.debit_account
                    if km.credit_account:
                        credit = km.credit_account
                    applied_score += km.weight
            except Exception:
                continue

        # Invoice/tax hints from text
        import re as _re
        inv_status = None
        if _re.search(r"T\d{13}", text_combined or ""):
            inv_status = "適格"
        else:
            if _re.search(r"(領収書|請求書|領収|請求)", text_combined or ""):
                inv_status = "非適格"
        t_lower = text_lower
        if any(k in t_lower for k in ("非課税", "不課税", "対象外")):
            inv_status = "非課税"
        elif any(k in t_lower for k in ("軽減", "8%", "８％", "8％")):
            inv_status = "軽減8%"
        elif any(k in t_lower for k in ("10%", "１０％", "10％")):
            inv_status = inv_status or "課税10%"

        # Persist
        doc_name = f"{source_path.name}#row{idx+1}"
        doc = Document(company_id=company_id, file_path=str(source_path.with_name(doc_name)), ocr_text=text_combined)
        session.add(doc)
        session.flush()

        auto = AutoResult(
            document_id=doc.id,
            date=date,
            amount=amount or 0,
            summary=summary,
            debit_account=debit,
            credit_account=credit,
            counterparty=counterparty,
            score=applied_score,
            invoice_status=inv_status,
        )
        session.add(auto)
        imported += 1

    return imported


def _run_llm_refine_paginated(session: Session, company_id: int, texts: dict) -> Optional[dict]:
    """Run LLM refinement page-by-page to avoid prompt size limits and combine results.

    - Splits both PDFテキストとYOMITOKUテキスト into page-level chunks heuristically.
    - Adds learned KeywordMapping as lightweight hints to each page payload.
    - Merges the first non-null date, most frequent amount, and majority-vote accounts.
    """
    import re as _re

    def _split_pages(s: str) -> list[str]:
        s = s or ""
        if "\f" in s:
            parts = [p.strip() for p in s.split("\f") if p.strip()]
            if parts:
                return parts
        # YOMITOKU: often markdown with headings like '## Page 1'
        pages = _re.split(r"\n\s*#{1,3}\s*Page\s*\d+.*\n", s)
        pages = [p.strip() for p in pages if p and p.strip()]
        if len(pages) > 1:
            return pages
        # Fallback: chunk by ~4000 chars
        chunk = 4000
        out: list[str] = []
        i = 0
        while i < len(s):
            out.append(s[i : i + chunk])
            i += chunk
        return [p.strip() for p in out if p.strip()]

    # PDF embedded text disabled; rely on YOMITOKU only
    pdf_all = ""
    yomi_all = (texts.get("text_yomitoku") or "").strip()

    pdf_pages = _split_pages(pdf_all)
    yomi_pages = _split_pages(yomi_all)
    # Align by index; if lengths differ, use closest available
    max_n = max(len(pdf_pages) or 1, len(yomi_pages) or 1)

    # Build hint lines from learned mappings
    hints_lines: list[str] = []
    try:
        for km in session.scalars(select(KeywordMapping).where(KeywordMapping.company_id == company_id)):
            if not km.keyword:
                continue
            hints_lines.append(
                f"- keyword='{km.keyword}' -> debit='{km.debit_account}' credit='{km.credit_account}' weight={km.weight}"
            )
    except Exception:
        pass
    hint_block = ("\n\n[学習ヒント]\n" + "\n".join(hints_lines)) if hints_lines else ""

    votes_amount: dict[int, int] = {}
    votes_debit: dict[str, int] = {}
    votes_credit: dict[str, int] = {}
    chosen_date: Optional[str] = None
    chosen_summary: Optional[str] = None

    for i in range(max_n):
        p_pdf = pdf_pages[i] if i < len(pdf_pages) else (pdf_pages[-1] if pdf_pages else "")
        p_yomi = yomi_pages[i] if i < len(yomi_pages) else (yomi_pages[-1] if yomi_pages else "")
        aug_yomi = (p_yomi + hint_block).strip()
        try:
            # Use YOMITOKU-only refinement
            r = refine_extraction_with_yomi("", aug_yomi, "")
        except Exception:
            r = None
        if not r:
            continue
        if not chosen_date and r.get("date"):
            chosen_date = r.get("date")
        if not chosen_summary and r.get("summary"):
            chosen_summary = r.get("summary")
        amt = r.get("amount")
        if isinstance(amt, int):
            votes_amount[amt] = votes_amount.get(amt, 0) + 1
        d = (r.get("debit_account") or "").strip()
        if d:
            votes_debit[d] = votes_debit.get(d, 0) + 1
        c = (r.get("credit_account") or "").strip()
        if c:
            votes_credit[c] = votes_credit.get(c, 0) + 1

    def _winner(dct: dict) -> Optional[Any]:
        return max(dct.items(), key=lambda kv: kv[1])[0] if dct else None

    if not (chosen_date or votes_amount or votes_debit or votes_credit or chosen_summary):
        return None
    return {
        "date": chosen_date,
        "amount": _winner(votes_amount),
        "summary": chosen_summary,
        "debit_account": _winner(votes_debit) or "",
        "credit_account": _winner(votes_credit) or "",
    }


def mark_check_later(session: Session, document_id: int) -> None:
    doc = session.get(Document, document_id)
    if not doc:
        raise ValueError("Document not found")
    doc.status = DocumentStatusEnum.CHECK_LATER.value
    session.add(Correction(document_id=document_id, action=CorrectionActionEnum.CHECK_LATER.value))


def delete_document(session: Session, document_id: int) -> None:
    doc = session.get(Document, document_id)
    if not doc:
        return
    session.delete(doc)


def _extract_tax_breakdown(text: str) -> list[dict]:
    """Extract 10%/8% tax-rate breakdown from YOMITOKU OCR text.

    Returns a list of dicts like {"rate": 10|8, "net": int|None, "tax": int|None, "gross": int|None}.
    Best-effort heuristics: looks for lines mentioning 10%/8% (軽減) and nearby
    lines with keywords like 対象/小計/金額 (net), 消費税/税額 (tax), 合計 (gross).
    """
    try:
        import re as _re

        def _to_half(s: str) -> str:
            tbl = str.maketrans({
                "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
                "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
                "％": "%", "，": ",", "．": ".",
            })
            return s.translate(tbl)

        def _nums_in(line: str) -> list[int]:
            line = _to_half(line)
            line = _re.sub(r"(?<=\d)\s+(?=\d)", "", line)
            out: list[int] = []
            for m in _re.finditer(r"([0-9]{1,3}(?:[,\s][0-9]{3})+|[0-9]+)", line):
                n = int(m.group(1).replace(",", "").replace(" ", ""))
                if n > 0:
                    out.append(n)
            return out

        def _rate_tag(s: str) -> int | None:
            ss = _to_half(s)
            low = ss.lower()
            is10 = ("10%" in low) or ("標準" in low)
            is8 = ("8%" in low) or ("軽減" in low)
            if is10 and not is8:
                return 10
            if is8 and not is10:
                return 8
            # ambiguous: return None
            return None

        lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
        found: dict[int, dict] = {}
        i = 0
        while i < len(lines):
            rate = _rate_tag(lines[i])
            if rate is None:
                i += 1
                continue
            rec = found.setdefault(rate, {"rate": rate, "net": None, "tax": None, "gross": None})
            # look ahead up to 2 lines for amounts with keywords
            for k in range(0, 3):
                if i + k >= len(lines):
                    break
                ln = lines[i + k]
                # stop if a different rate block clearly starts
                other = _rate_tag(ln)
                if other is not None and other != rate and k > 0:
                    break
                nums = _nums_in(ln)
                if not nums:
                    continue
                n = nums[-1]  # prefer the last number on the line
                low = _to_half(ln).lower()
                if ("税" in low) or ("消費税" in low):
                    rec["tax"] = rec["tax"] or n
                elif ("合計" in low) or ("計" in low and "明細" not in low):
                    rec["gross"] = rec["gross"] or n
                elif any(k in low for k in ("対象", "小計", "金額", "売上", "仕入")):
                    rec["net"] = rec["net"] or n
            i += 1

        # compute missing values
        out: list[dict] = []
        for rate, rec in sorted(found.items()):
            net, tax, gross = rec.get("net"), rec.get("tax"), rec.get("gross")
            try:
                if net is not None and tax is not None and gross is None:
                    gross = net + tax
                elif gross is not None and tax is not None and net is None:
                    net = max(gross - tax, 0)
                elif gross is not None and net is not None and tax is None:
                    tax = max(gross - net, 0)
            except Exception:
                pass
            if any(v is not None for v in (net, tax, gross)):
                out.append({"rate": rate, "net": net, "tax": tax, "gross": gross})
        return out
    except Exception:
        return []


def _build_entries_for_document(doc: Document) -> list[JournalEntry]:
    """Create one or more JournalEntry rows for a document.

    - If 10%/8% breakdown is detected, emit one row per rate with tax_category/tax_amount set on the debit side.
    - Otherwise, emit a single row with no tax fields.
    """
    entries: list[JournalEntry] = []
    ar = doc.auto_result
    if not ar:
        return entries

    from datetime import datetime as _dt

    def _tax_cat(rate: int) -> str:
        return "課税仕入8%(軽)" if rate == 8 else "課税仕入10%"

    parts = _extract_tax_breakdown(doc.ocr_text or "")
    if parts:
        total = len(parts)
        for idx, p in enumerate(parts):
            try:
                rate = int(p.get("rate") or 0)
            except Exception:
                rate = 0
            net = p.get("net")
            tax = p.get("tax") or 0
            if not net:
                # as a fallback, prefer gross if available
                gross = p.get("gross")
                if gross is None:
                    continue
                net = max(int(gross) - int(tax or 0), 0)
            amount_base = abs(int(net))
            tax_amt = abs(int(tax or 0))
            # 識別フラグ: 複数行のときは 2110/2100/2101 を付与
            if total <= 1:
                ident = "2111"
            elif idx == 0:
                ident = "2110"
            elif idx == total - 1:
                ident = "2101"
            else:
                ident = "2100"
            entry = JournalEntry(
                identifier_flag=ident,
                voucher_number="",
                transaction_date=ar.date or _dt.now().strftime("%Y/%m/%d"),
                debit=AccountSide(
                    account=ar.debit_account or "",
                    amount=amount_base,
                    # 税区分(H) 必須: 請求書区分が「適格」なら内税表記
                    tax_category=(
                        ("課対仕入内10%適格" if getattr(ar, "invoice_status", None) == "適格" else "課対仕入10%")
                        if rate in (8, 10)
                        else ("課対仕入内10%適格" if getattr(ar, "invoice_status", None) == "適格" else "課対仕入10%")
                    ),
                    tax_amount=tax_amt,
                ),
                credit=AccountSide(
                    account=ar.credit_account or "",
                    amount=amount_base,
                    tax_category=(("課対仕入内10%適格" if getattr(ar, "invoice_status", None) == "適格" else "課対仕入10%") if rate in (8, 10) else ("課対仕入内10%適格" if getattr(ar, "invoice_status", None) == "適格" else "課対仕入10%")),
                    tax_amount=0,
                ),
                summary=(ar.summary or "")[:64],
                reference_number="",
                memo="",
            )
            entries.append(entry)

    if not entries:
        # Fallback: single-line entry without tax fields
        entry = JournalEntry(
            identifier_flag="2111",
            voucher_number="",
            transaction_date=ar.date or _dt.now().strftime("%Y/%m/%d"),
            debit=AccountSide(
                account=ar.debit_account or "",
                amount=abs(ar.amount or 0),
                tax_category=("課対仕入内10%適格" if getattr(ar, "invoice_status", None) == "適格" else "課対仕入10%"),
                tax_amount=0,
            ),
            credit=AccountSide(
                account=ar.credit_account or "",
                amount=abs(ar.amount or 0),
                tax_category=("課対仕入内10%適格" if getattr(ar, "invoice_status", None) == "適格" else "課対仕入10%"),
                tax_amount=0,
            ),
            summary=(ar.summary or "")[:64],
            reference_number="",
            memo="",
        )
        entries.append(entry)

    return entries

def confirm_document(
    session: Session,
    document_id: int,
    date: Optional[str],
    amount: Optional[int],
    summary: Optional[str],
    debit: Optional[str],
    credit: Optional[str],
    debit_sub: Optional[str] = None,
    credit_sub: Optional[str] = None,
    invoice_status: Optional[str] = None,
) -> Path:
    doc = session.get(Document, document_id)
    if not doc or not doc.auto_result:
        raise ValueError("Document not found")

    # Apply corrections to auto result
    # Be defensive about legacy SQLite schemas that might miss newer columns
    def _table_has(col: str) -> bool:
        try:
            bind = session.get_bind()
            name = getattr(bind.dialect, "name", "") if bind is not None else ""
            if name == "sqlite":
                rows = bind.exec_driver_sql("PRAGMA table_info('auto_results')").all()  # type: ignore[attr-defined]
                cols = {r[1] for r in rows}
                return col in cols
        except Exception:
            pass
        # For non-SQLite or on failure, assume column exists (modern schema)
        return True
    if date is not None:
        doc.auto_result.date = date
    if amount is not None:
        doc.auto_result.amount = amount
    if summary is not None:
        doc.auto_result.summary = summary
    if debit is not None:
        doc.auto_result.debit_account = debit
    if credit is not None:
        doc.auto_result.credit_account = credit
    if debit_sub is not None and _table_has("debit_subaccount"):
        doc.auto_result.debit_subaccount = debit_sub or None
    if credit_sub is not None and _table_has("credit_subaccount"):
        doc.auto_result.credit_subaccount = credit_sub or None
    if invoice_status is not None and _table_has("invoice_status"):
        doc.auto_result.invoice_status = invoice_status or None

    doc.status = DocumentStatusEnum.CONFIRMED.value
    session.add(
        Correction(
            document_id=document_id,
            action=CorrectionActionEnum.OK.value,
            corrected_date=doc.auto_result.date,
            corrected_amount=doc.auto_result.amount,
            corrected_summary=doc.auto_result.summary,
            debit_account=doc.auto_result.debit_account,
            credit_account=doc.auto_result.credit_account,
        )
    )

    # Learning: update or insert keyword mappings for top keywords (counterparty words)
    text = (doc.ocr_text or "").lower()
    candidates = []
    # pick up to 5 longest words that are alphabetic or Katakana-like chunks
    import re

    for m in re.finditer(r"[A-Za-z0-9ァ-ンｧ-ﾝﾞﾟ一-龥]{3,}", text):
        w = m.group(0)[:32].lower()
        if any(k in w for k in ("合計", "金額", "領収", "請求", "明細")):
            continue
        candidates.append(w)
    candidates = sorted(set(candidates), key=len, reverse=True)[:5]

    for kw in candidates:
        try:
            km = KeywordMapping(
                company_id=doc.company_id,
                keyword=kw,
                debit_account=doc.auto_result.debit_account,
                credit_account=doc.auto_result.credit_account,
                weight=1,
            )
            session.add(km)
            session.flush()
        except IntegrityError:
            session.rollback()
            # increment weight if exists
            existing = session.scalar(
                select(KeywordMapping).where(
                    KeywordMapping.company_id == doc.company_id, KeywordMapping.keyword == kw
                )
            )
            if existing:
                existing.debit_account = doc.auto_result.debit_account or existing.debit_account
                existing.credit_account = doc.auto_result.credit_account or existing.credit_account
                existing.weight += 1
                session.add(existing)

    # Export a single-line CSV per confirmed document with its date
    settings = get_settings()
    date_for_name = (doc.auto_result.date or datetime.now().strftime("%Y/%m/%d")).replace("/", "")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.output_dir / f"{date_for_name}_yayoi.csv"

    entries = _build_entries_for_document(doc)
    exporter = JournalExporter(entries=entries)
    exporter.save(out_path, encoding="utf-8", bom=True)

    # Duplicate and archive stamped PDF copies (disabled for test environment)
    # try:
    #     _archive_stamped_pdf(session, doc)
    # except Exception:
    #     # Non-fatal: continue even if archiving fails
    #     pass
    return out_path


def confirm_document_multi(
    session: Session,
    document_id: int,
    rows: list[dict[str, object]],
) -> Path:
    """Confirm a document with multiple journal rows and export a multi-line Yayoi CSV.

    - Updates the document's AutoResult from the first row (for consistency with lists and logs).
    - Marks the document as CONFIRMED.
    - Exports all provided rows as individual lines in one CSV.
    """
    doc = session.get(Document, document_id)
    if not doc or not doc.auto_result:
        raise ValueError("Document not found")
    if not rows:
        raise ValueError("No rows provided")

    first = rows[0]

    def _get(d: dict[str, object], k: str) -> str | None:
        v = d.get(k)
        return str(v).strip() if isinstance(v, str) else (str(v).strip() if v is not None else None)

    # Update auto_result from the first row
    try:
        if first.get("date") is not None:
            doc.auto_result.date = _get(first, "date")
        if first.get("amount") is not None:
            try:
                doc.auto_result.amount = int(first.get("amount") or 0)
            except Exception:
                pass
        if first.get("summary") is not None:
            doc.auto_result.summary = _get(first, "summary") or ""
        if first.get("debit_account") is not None:
            doc.auto_result.debit_account = _get(first, "debit_account") or ""
        if first.get("credit_account") is not None:
            doc.auto_result.credit_account = _get(first, "credit_account") or ""
    except Exception:
        pass

    doc.status = DocumentStatusEnum.CONFIRMED.value
    session.add(doc)

    # Replace existing lines for this document
    try:
        from sqlalchemy import delete as _delete
        session.execute(_delete(AutoResultLine).where(AutoResultLine.document_id == doc.id))
    except Exception:
        pass

    # Build entries from rows
    from datetime import datetime as _dt
    try:
        base_date = doc.auto_result.date or _dt.now().strftime("%Y/%m/%d")
    except Exception:
        base_date = _dt.now().strftime("%Y/%m/%d")

    entries: list[JournalEntry] = []
    total_rows = len(rows)
    for i, r in enumerate(rows):
        try:
            dt = _get(r, "date") or base_date
            debit = _get(r, "debit_account") or ""
            credit = _get(r, "credit_account") or ""
            try:
                amt = abs(int(r.get("amount") or 0))
            except Exception:
                amt = abs(int(doc.auto_result.amount or 0))
            summary = _get(r, "summary") or (doc.auto_result.summary or "")
            # Persist line
            try:
                session.add(
                    AutoResultLine(
                        document_id=doc.id,
                        line_no=i + 1,
                        date=dt,
                        amount=amt,
                        summary=summary,
                        debit_account=debit,
                        credit_account=credit,
                    )
                )
            except Exception:
                pass
            # 識別フラグ: 1行=2111、複数行は 2110/2100/2101
            if total_rows <= 1:
                ident = "2111"
            elif i == 0:
                ident = "2110"
            elif i == total_rows - 1:
                ident = "2101"
            else:
                ident = "2100"
            entry = JournalEntry(
                identifier_flag=ident,
                voucher_number="",
                transaction_date=dt,
                debit=AccountSide(
                    account=debit,
                    amount=amt,
                    tax_category=("課対仕入内10%適格" if getattr(doc.auto_result, "invoice_status", None) == "適格" else "課対仕入10%"),
                    tax_amount=0,
                ),
                credit=AccountSide(account=credit, amount=amt),
                summary=summary[:64],
                reference_number="",
                memo="",
            )
            entries.append(entry)
        except Exception:
            continue

    if not entries:
        raise ValueError("No valid rows to export")

    # Save multi-line CSV
    settings = get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = settings.output_dir / f"{ts}_yayoi.csv"
    exporter = JournalExporter(entries=entries)
    exporter.save(out_path, encoding="utf-8", bom=True)

    # Archive stamped PDF copies (disabled for test environment)
    # try:
    #     _archive_stamped_pdf(session, doc)
    # except Exception:
    #     pass

    # Record a single correction entry (multi)
    try:
        session.add(
            Correction(
                document_id=doc.id,
                action=CorrectionActionEnum.OK.value,
                corrected_date=doc.auto_result.date,
                corrected_amount=doc.auto_result.amount,
                corrected_summary=doc.auto_result.summary,
                debit_account=doc.auto_result.debit_account,
                credit_account=doc.auto_result.credit_account,
            )
        )
    except Exception:
        pass

    return out_path


def list_documents(session: Session, company_id: int, status: Optional[str] = None) -> list[Document]:
    stmt = select(Document).where(Document.company_id == company_id)
    if status:
        stmt = stmt.where(Document.status == status)
    stmt = stmt.order_by(Document.created_at.desc())
    return list(session.scalars(stmt))


def export_confirmed_csv(
    session: Session,
    company_id: int,
    encoding: str = "utf-8",
    bom: bool = False,
    output_dir: Path | None = None,
) -> Path:
    """Export all confirmed documents for a company into a single Yayoi CSV.

    Returns the output file path.
    """
    docs = list_documents(session, company_id, status=DocumentStatusEnum.CONFIRMED.value)
    if not docs:
        raise ValueError("No confirmed documents to export")

    entries: list[JournalEntry] = []
    included_ids: list[int] = []
    from sqlalchemy import select as _select
    for d in docs:
        if not d.auto_result:
            continue
        # If multi-line entries exist for this document, use them; otherwise fallback
        try:
            lines = list(session.scalars(_select(AutoResultLine).where(AutoResultLine.document_id == d.id)))
        except Exception:
            lines = []
        if lines:
            from datetime import datetime as _dt
            total = len(lines)
            for i, ln in enumerate(lines):
                try:
                    dt = ln.date or (d.auto_result.date or _dt.now().strftime("%Y/%m/%d"))
                    amt = abs(int(ln.amount or 0))
                except Exception:
                    dt = d.auto_result.date or _dt.now().strftime("%Y/%m/%d")
                    amt = abs(int(d.auto_result.amount or 0))
                if total <= 1:
                    ident = "2111"
                elif i == 0:
                    ident = "2110"
                elif i == total - 1:
                    ident = "2101"
                else:
                    ident = "2100"
                entries.append(
                    JournalEntry(
                        identifier_flag=ident,
                        voucher_number="",
                        transaction_date=dt,
                        debit=AccountSide(
                            account=(ln.debit_account or d.auto_result.debit_account or ""),
                            amount=amt,
                            tax_category=("課対仕入内10%適格" if getattr(d.auto_result, "invoice_status", None) == "適格" else "課対仕入10%"),
                            tax_amount=0,
                        ),
                        credit=AccountSide(account=(ln.credit_account or d.auto_result.credit_account or ""), amount=amt),
                        summary=((ln.summary or d.auto_result.summary or "")[:64]),
                        reference_number="",
                        memo="",
                    )
                )
            included_ids.append(int(d.id))
        else:
            entries.extend(_build_entries_for_document(d))
            included_ids.append(int(d.id))

    settings = get_settings()
    target_dir = output_dir or settings.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = target_dir / f"{ts}_yayoi.csv"
    exporter = JournalExporter(entries=entries)
    exporter.save(out_path, encoding=encoding, bom=bom)

    # Write manifest for this export (to enable later restore)
    try:
        import json as _json
        manifest = {
            "csv": str(out_path),
            "exported_at": datetime.now().isoformat(),
            "company_id": int(company_id),
            "document_ids": included_ids,
        }
        man_path = out_path.with_suffix("").with_name(out_path.stem).with_suffix(".json")
        # If stem has .csv, replace with .json; above ensures .json alongside
        with open(man_path, "w", encoding="utf-8") as fh:
            _json.dump(manifest, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Mark exported docs as EXPORTED
    try:
        idset = set(included_ids)
        for d in docs:
            if int(d.id) in idset:
                d.status = DocumentStatusEnum.EXPORTED.value
                session.add(d)
    except Exception:
        pass
    return out_path


# Company settings helpers
def get_company_settings(session: Session, company_id: int) -> CompanySetting:
    cs = session.scalar(select(CompanySetting).where(CompanySetting.company_id == company_id))
    if cs:
        return cs
    cs = CompanySetting(company_id=company_id, default_output_dir=None, archive_base_dir=None)
    session.add(cs)
    session.flush()
    return cs


def update_company_settings(
    session: Session, company_id: int, default_output_dir: Optional[str], archive_base_dir: Optional[str]
) -> CompanySetting:
    cs = session.scalar(select(CompanySetting).where(CompanySetting.company_id == company_id))
    if not cs:
        cs = CompanySetting(
            company_id=company_id,
            default_output_dir=default_output_dir,
            archive_base_dir=archive_base_dir,
        )
        session.add(cs)
    else:
        cs.default_output_dir = default_output_dir
        cs.archive_base_dir = archive_base_dir
        session.add(cs)
    session.flush()
    return cs


def _safe_name(name: str) -> str:
    import re

    s = name.strip().replace(" ", "_")
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    return s[:80] or "unknown"


def _stamp_pdf(input_pdf: Path, label: str, output_pdf: Path) -> None:
    from PyPDF2 import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import Color

    reader = PdfReader(str(input_pdf))
    writer = PdfWriter()
    first_page = reader.pages[0]
    # Get size
    media_box = first_page.mediabox
    width = float(media_box.width)
    height = float(media_box.height)

    # Create overlay PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmpf:
        overlay_path = Path(tmpf.name)
    try:
        c = canvas.Canvas(str(overlay_path), pagesize=(width, height))
        c.setFillColor(Color(0, 0, 0, alpha=0.6))
        c.setFont("Helvetica", 12)
        margin = 36
        lines = label.splitlines()
        y = height - margin
        for line in lines:
            c.drawString(margin, y, line[:180])
            y -= 14
        c.save()

        from PyPDF2 import PdfReader as _R

        overlay_reader = _R(str(overlay_path))
        overlay_page = overlay_reader.pages[0]

        # Merge overlay onto first page, copy others
        new_first = first_page
        try:
            new_first.merge_page(overlay_page)  # PyPDF2>=2
        except Exception:
            new_first.mergePage(overlay_page)  # legacy
        writer.add_page(new_first)
        for i in range(1, len(reader.pages)):
            writer.add_page(reader.pages[i])

        with output_pdf.open("wb") as out_f:
            writer.write(out_f)
    finally:
        try:
            overlay_path.unlink(missing_ok=True)
        except Exception:
            pass


def _archive_stamped_pdf(session: Session, doc: Document) -> None:
    # Resolve archive base directory from company settings
    cs = get_company_settings(session, doc.company_id)
    settings = get_settings()
    base = Path(cs.archive_base_dir) if cs.archive_base_dir else (settings.data_dir / "archive")
    base.mkdir(parents=True, exist_ok=True)

    ar = doc.auto_result
    if not ar:
        return
    # Build label text
    date = ar.date or datetime.now().strftime("%Y/%m/%d")
    label = (
        f"会社: {session.get(Company, doc.company_id).name}\n"
        f"日付: {date}  金額: {ar.amount or 0}\n"
        f"科目: 借方={ar.debit_account} 貸方={ar.credit_account}\n"
        f"摘要: {ar.summary}"
    )

    src = Path(doc.file_path)
    # Prepare stamped temp copy
    date_compact = date.replace("/", "-")
    base_name = f"{date_compact}_{_safe_name(Path(doc.file_path).stem)}_labeled.pdf"

    with tempfile.TemporaryDirectory() as td:
        tmp_out = Path(td) / base_name
        try:
            _stamp_pdf(src, label, tmp_out)
        except Exception:
            # If stamping fails, fall back to just copying
            shutil.copy2(src, tmp_out)

        # 1) By date: base/by_date/YYYY/YYYY-MM/
        dt = datetime.strptime(date.replace("/", "-"), "%Y-%m-%d")
        by_date_dir = base / "by_date" / f"{dt.year}" / f"{dt.year}-{dt.month:02d}"
        by_date_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_out, by_date_dir / base_name)

        # 2) By account: one copy per involved account name (deduplicated)
        accounts = {_safe_name(ar.debit_account or ""), _safe_name(ar.credit_account or "")} - {""}
        for acc in accounts:
            acc_dir = base / "by_account" / acc / f"{dt.year}"
            acc_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_out, acc_dir / base_name)


# Account settings helpers (subaccounts and default summaries per account)
def get_account_settings(session: Session, company_id: int) -> list[AccountSetting]:
    return list(session.scalars(select(AccountSetting).where(AccountSetting.company_id == company_id)))


def upsert_account_setting(
    session: Session,
    company_id: int,
    account_name: str,
    subaccounts: list[str] | None,
    summaries: list[str] | None,
) -> AccountSetting:
    row = session.scalar(
        select(AccountSetting).where(
            AccountSetting.company_id == company_id, AccountSetting.account_name == account_name
        )
    )
    import json as _json

    subs_json = _json.dumps(subaccounts or [], ensure_ascii=False)
    sums_json = _json.dumps(summaries or [], ensure_ascii=False)
    if not row:
        row = AccountSetting(
            company_id=company_id,
            account_name=account_name,
            subaccounts_json=subs_json,
            summaries_json=sums_json,
        )
        session.add(row)
    else:
        row.subaccounts_json = subs_json
        row.summaries_json = sums_json
        session.add(row)
    session.flush()
    return row


def list_tax_categories(session: Session) -> list[TaxCategory]:
    return list(session.scalars(select(TaxCategory).where(TaxCategory.active == 1).order_by(TaxCategory.name)))


# Duplicate detection helpers
def _doc_auto_dict(doc: Document) -> dict:
    ar = doc.auto_result
    return {
        "id": doc.id,
        "file_path": doc.file_path,
        "status": doc.status,
        "auto": {
            "date": ar.date if ar else None,
            "amount": ar.amount if ar else None,
            "summary": ar.summary if ar else "",
            "debit_account": ar.debit_account if ar else "",
            "credit_account": ar.credit_account if ar else "",
            "counterparty": ar.counterparty if ar else "",
            "invoice_status": getattr(ar, "invoice_status", None) if ar else None,
        },
    }


def find_duplicate_candidates(session: Session, company_id: int) -> list[dict]:
    """Return a list of potential duplicate matches for new (unconfirmed) documents.

    Strategy: for each unconfirmed document with an auto_result, look for other
    documents in the same company with the same date and amount. A simple score
    is computed and only matches with score>=2 are returned.
    """
    out: list[dict] = []
    from sqlalchemy import and_

    # Unconfirmed docs as new targets
    new_docs = list(
        session.scalars(
            select(Document).where(
                Document.company_id == company_id,
                Document.status == DocumentStatusEnum.UNCONFIRMED.value,
            )
        )
    )
    for nd in new_docs:
        nd_ar = nd.auto_result
        if not nd_ar:
            continue
        nd_date = (nd_ar.date or "").strip()
        nd_amt = nd_ar.amount
        if not nd_date or nd_amt is None:
            continue
        # Candidate set: same amount and same date
        cands = list(
            session.scalars(
                select(Document).where(
                    Document.company_id == company_id,
                    Document.id != nd.id,
                )
            )
        )
        matches: list[dict] = []
        for od in cands:
            oa = od.auto_result
            if not oa:
                continue
            score = 0
            if oa.amount == nd_amt:
                score += 1
            if (oa.date or "").strip() == nd_date:
                score += 1
            # lightweight summary hint
            ns = (nd_ar.summary or "").strip().lower()
            os = (oa.summary or "").strip().lower()
            if ns and os and (ns[:10] == os[:10]):
                score += 1
            if score >= 2:
                matches.append({"score": score, "document": _doc_auto_dict(od)})
        if matches:
            # sort best first
            matches.sort(key=lambda m: m.get("score", 0), reverse=True)
            out.append({"new": _doc_auto_dict(nd), "matches": matches})
    return out


def get_document_detail(session: Session, document_id: int) -> dict:
    doc = session.get(Document, document_id)
    if not doc:
        raise ValueError("Document not found")
    return _doc_auto_dict(doc)


# Global rule helpers (initial default auto-journal rules)
def list_global_rules(session: Session) -> list[GlobalRule]:
    return list(
        session.scalars(
            select(GlobalRule).order_by(GlobalRule.enabled.desc(), GlobalRule.priority.desc(), GlobalRule.id.asc())
        )
    )


def upsert_global_rule(
    session: Session,
    rule_id: Optional[int],
    keyword: str,
    debit_account: str,
    credit_account: str,
    priority: int = 0,
    enabled: bool = True,
) -> GlobalRule:
    if rule_id:
        row = session.get(GlobalRule, rule_id)
    else:
        row = None
    if not row:
        row = GlobalRule(
            keyword=keyword,
            debit_account=debit_account,
            credit_account=credit_account,
            priority=priority,
            enabled=1 if enabled else 0,
        )
        session.add(row)
    else:
        row.keyword = keyword
        row.debit_account = debit_account
        row.credit_account = credit_account
        row.priority = priority
        row.enabled = 1 if enabled else 0
        session.add(row)
    session.flush()
    return row


def delete_global_rule(session: Session, rule_id: int) -> None:
    row = session.get(GlobalRule, rule_id)
    if row:
        session.delete(row)


# Import account names from a PDF into company AccountSetting rows
def import_accounts_pdf_to_company(session: Session, company_id: int, pdf_path: Path) -> int:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        raise RuntimeError("pdfminer.six is required to parse the PDF. Please install requirements.")

    text = extract_text(str(pdf_path)) or ""
    lines = [l.strip() for l in text.splitlines()]
    count = 0

    import re

    for ln in lines:
        if not ln:
            continue
        if any(k in ln for k in ("勘定科目", "一覧", "税区分", "部門", "補助", "摘要", "コード", "Code")):
            continue
        name = None
        m = re.match(r"^\s*(\d{2,4})\s+(.+)$", ln)
        if m:
            name = m.group(2).strip()
        else:
            name = ln.strip()
        # basic sanity
        if len(name) < 2 or len(name) > 32:
            continue
        if not re.search(r"[\w一-龥ぁ-んァ-ヴ]", name):
            continue
        try:
            upsert_account_setting(session, company_id, name, None, None)
            count += 1
        except Exception:
            # ignore duplicates or parsing oddities
            pass
    session.flush()
    return count


def import_accounts_from_text(session: Session, company_id: int, text: str) -> int:
    """Import account names from a plain text list.

    Expected patterns per line (examples):
      - 現金 GENKIN 100
      - 売上高 URIAGE 700
      - 単なる見出しや空行はスキップ
    Only the leading Japanese name is stored as account_name.
    """
    lines = [l.strip() for l in (text or "").splitlines()]
    if not lines:
        return 0
    count = 0
    import re

    # Heuristic: heading keywords to skip
    skip_kw = ("資産の部", "流動資産", "固定資産", "繰延資産", "諸口", "負債の部", "流動負債", "固定負債", "純資産の部", "収益の部", "売上高", "売上原価", "販売費および一般管理費", "営業外", "特別損益", "など")

    seen: set[str] = set()
    for ln in lines:
        if not ln:
            continue
        if any(k in ln for k in skip_kw):
            continue
        # match: <japanese name> <UPPERCODE> [number]
        m = re.match(r"^([\u3040-\u30ff\u3400-\u9fffA-Za-z0-9（）・\-]+?)\s+[A-Z\-]+(?:\s+\d+)?\s*$", ln)
        if m:
            name = m.group(1).strip()
        else:
            # If no code token, skip (likely a heading or noise)
            continue
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            upsert_account_setting(session, company_id, name, None, None)
            count += 1
        except Exception:
            pass
    session.flush()
    return count


def import_account_master_from_text(session: Session, text: str) -> int:
    """Import global master account names from pasted plain text.

    Pattern: 日本語名称 + 英大文字コード + 任意の番号 を優先。重複はスキップ。
    """
    lines = [l.strip() for l in (text or "").splitlines()]
    if not lines:
        return 0
    import re
    skip_kw = ("資産の部", "流動資産", "固定資産", "繰延資産", "諸口", "負債の部", "流動負債", "固定負債", "純資産の部", "収益の部", "売上高", "売上原価", "販売費および一般管理費", "営業外", "特別損益", "など")
    seen = set()
    count = 0
    for ln in lines:
        if not ln or any(k in ln for k in skip_kw):
            continue
        m = re.match(r"^([\u3040-\u30ff\u3400-\u9fffA-Za-z0-9（）・\-]+?)\s+([A-Z\-]+)(?:\s+\d+)?\s*$", ln)
        if not m:
            continue
        name = m.group(1).strip()
        code = m.group(2).strip()
        if not name:
            continue
        key = (name, code)
        if key in seen:
            continue
        seen.add(key)
        try:
            row = session.scalar(select(AccountMaster).where(AccountMaster.name == name))
            if not row:
                session.add(AccountMaster(name=name, code=code))
                count += 1
        except Exception:
            pass
    session.flush()
    return count


def list_account_master(session: Session) -> list[AccountMaster]:
    return list(session.scalars(select(AccountMaster).order_by(AccountMaster.name)))

# Keyword mapping upsert for company-level NL instructions
def upsert_keyword_mapping(
    session: Session,
    company_id: int,
    keyword: str,
    debit_account: str,
    credit_account: str,
    weight: int = 1,
) -> KeywordMapping:
    row = session.scalar(
        select(KeywordMapping).where(
            KeywordMapping.company_id == company_id, KeywordMapping.keyword == keyword
        )
    )
    if not row:
        row = KeywordMapping(
            company_id=company_id,
            keyword=keyword,
            debit_account=debit_account,
            credit_account=credit_account,
            weight=weight,
        )
        session.add(row)
    else:
        row.debit_account = debit_account or row.debit_account
        row.credit_account = credit_account or row.credit_account
        row.weight = max(weight, row.weight)
        session.add(row)
    session.flush()
    return row
