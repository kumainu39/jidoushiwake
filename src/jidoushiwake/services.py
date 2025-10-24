from __future__ import annotations

from datetime import datetime
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_session, engine, Base
from .models import (
    AutoResult,
    Company,
    Correction,
    CorrectionActionEnum,
    Document,
    DocumentStatusEnum,
    KeywordMapping,
    CompanySetting,
    CompanyLLMSetting,
    LLMLog,
    AccountSetting,
)
from .ocr_extract import extract_text_from_pdf, extract_journal_data, extract_text_both
from .yayoi_exporter import AccountSide, JournalEntry, JournalExporter
from .llm_client import refine_extraction


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
    except Exception:
        # Never block app startup due to migration helper; the API will surface
        # errors if the column truly cannot be added (e.g., permissions).
        pass


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
                device=c_llm.device,
                n_gpu_layers=c_llm.n_gpu_layers,
                n_threads=c_llm.n_threads,
                lora_path=c_llm.lora_path,
                prompt_template=c_llm.prompt_template,
            )
            with temporary_config(cfg):
                # Feed YOMITOKU first (if available) in the image OCR slot to prioritize it.
                _pdf_for_llm = texts.get("text_pdf") or ""
                _img_for_llm = ( (texts.get("text_yomitoku") or "") + "\n\n" + (texts.get("text_paddle") or "") ).strip()
                llm_ref = refine_extraction(_pdf_for_llm, _img_for_llm)
        else:
            _pdf_for_llm = texts.get("text_pdf") or ""
            _img_for_llm = ( (texts.get("text_yomitoku") or "") + "\n\n" + (texts.get("text_paddle") or "") ).strip()
            llm_ref = refine_extraction(_pdf_for_llm, _img_for_llm)
        if llm_ref:
            parsed.date = llm_ref.get("date") or parsed.date
            try:
                amt = llm_ref.get("amount")
                parsed.amount = int(amt) if amt is not None else parsed.amount
            except Exception:
                pass
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
                    device=(c_llm.device if c_llm and c_llm.use_override else None),
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

    # Apply per-company keyword mappings to refine accounts
    debit = parsed.debit_account
    credit = parsed.credit_account
    applied_score = 0
    for km in session.scalars(select(KeywordMapping).where(KeywordMapping.company_id == company_id)):
        if km.keyword and (km.keyword.lower() in text.lower()):
            if km.debit_account:
                debit = km.debit_account
            if km.credit_account:
                credit = km.credit_account
            applied_score += km.weight

    # Detect qualified invoice registration number (T + 13 digits)
    import re as _re
    inv_status = "適格" if _re.search(r"T\d{13}", text_combined) else None

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
    if debit_sub is not None:
        doc.auto_result.debit_subaccount = debit_sub or None
    if credit_sub is not None:
        doc.auto_result.credit_subaccount = credit_sub or None
    if invoice_status is not None:
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

    entry = JournalEntry(
        identifier_flag="****",
        voucher_number="1",
        transaction_date=doc.auto_result.date or datetime.now().strftime("%Y/%m/%d"),
        debit=AccountSide(account=doc.auto_result.debit_account, amount=abs(doc.auto_result.amount or 0)),
        credit=AccountSide(account=doc.auto_result.credit_account, amount=abs(doc.auto_result.amount or 0)),
        summary=(doc.auto_result.summary or "")[:64],
        reference_number=str(Path(doc.file_path).stem)[:10],
        memo=str(Path(doc.file_path).name)[:180],
    )

    exporter = JournalExporter(entries=[entry])
    exporter.save(out_path, encoding="utf-8")

    # Duplicate and archive stamped PDF copies
    try:
        _archive_stamped_pdf(session, doc)
    except Exception:
        # Non-fatal: continue even if archiving fails
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
    for d in docs:
        if not d.auto_result:
            continue
        ar = d.auto_result
        entry = JournalEntry(
            identifier_flag="****",
            voucher_number="1",
            transaction_date=ar.date or datetime.now().strftime("%Y/%m/%d"),
            debit=AccountSide(account=ar.debit_account or "", amount=abs(ar.amount or 0)),
            credit=AccountSide(account=ar.credit_account or "", amount=abs(ar.amount or 0)),
            summary=(ar.summary or "")[:64],
            reference_number=str(Path(d.file_path).stem)[:10],
            memo=str(Path(d.file_path).name)[:180],
        )
        entries.append(entry)

    settings = get_settings()
    target_dir = output_dir or settings.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = target_dir / f"{ts}_yayoi.csv"
    exporter = JournalExporter(entries=entries)
    exporter.save(out_path, encoding=encoding, bom=bom)
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
