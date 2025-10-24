from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..db import get_session
from ..models import Company, DocumentStatusEnum
from ..services import (
    confirm_document,
    delete_document,
    import_pdf,
    init_db,
    list_companies,
    list_documents,
    mark_check_later,
    ensure_company,
    export_confirmed_csv,
    get_company_settings,
    update_company_settings,
    get_account_settings,
    upsert_account_setting,
)
from ..ocr_master import OCRMasterSample, OCRMasterSettings
from ..llm_client import LLMConfig, set_config as set_llm_config
from ..db import get_session
from sqlalchemy import select, func


app = FastAPI(title="Jidoushiwake API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompanyIn(BaseModel):
    name: str


class DocumentOut(BaseModel):
    id: int
    file_path: str
    status: str
    auto: Optional[dict]


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/companies", response_model=list[CompanyIn])
def api_list_companies():
    with get_session() as s:
        items = list_companies(s)
        return [CompanyIn(name=c.name) for c in items]


@app.post("/companies", response_model=CompanyIn)
def api_create_company(payload: CompanyIn):
    with get_session() as s:
        c = ensure_company(s, payload.name)
        return CompanyIn(name=c.name)


@app.post("/documents/import", response_model=DocumentOut)
async def api_import_pdf(company_name: str = Form(...), file: UploadFile = File(...)):
    # Save uploaded to a temp storage under data/uploads
    contents = await file.read()
    uploads_dir = Path("data") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / file.filename
    dest.write_bytes(contents)

    with get_session() as s:
        company = ensure_company(s, company_name)
        doc = import_pdf(s, company.id, dest)
        auto = doc.auto_result
        return DocumentOut(
            id=doc.id,
            file_path=doc.file_path,
            status=doc.status,
            auto={
                "date": auto.date if auto else None,
                "amount": auto.amount if auto else None,
                "summary": auto.summary if auto else "",
                "debit_account": auto.debit_account if auto else "",
                "debit_subaccount": getattr(auto, "debit_subaccount", None) if auto else None,
                "credit_account": auto.credit_account if auto else "",
                "credit_subaccount": getattr(auto, "credit_subaccount", None) if auto else None,
                "counterparty": auto.counterparty if auto else "",
                "invoice_status": getattr(auto, "invoice_status", None) if auto else None,
            },
        )


@app.get("/documents", response_model=list[DocumentOut])
def api_list_documents(company_name: str, status: Optional[str] = None):
    with get_session() as s:
        company = ensure_company(s, company_name)
        docs = list_documents(s, company.id, status)
        items: list[DocumentOut] = []
        for d in docs:
            auto = d.auto_result
            items.append(
                DocumentOut(
                    id=d.id,
                    file_path=d.file_path,
                    status=d.status,
                    auto={
                        "date": auto.date if auto else None,
                        "amount": auto.amount if auto else None,
                        "summary": auto.summary if auto else "",
                        "debit_account": auto.debit_account if auto else "",
                        "debit_subaccount": getattr(auto, "debit_subaccount", None) if auto else None,
                        "credit_account": auto.credit_account if auto else "",
                        "credit_subaccount": getattr(auto, "credit_subaccount", None) if auto else None,
                        "counterparty": auto.counterparty if auto else "",
                        "invoice_status": getattr(auto, "invoice_status", None) if auto else None,
                    },
                )
            )
        return items


class ConfirmIn(BaseModel):
    date: Optional[str] = None
    amount: Optional[int] = None
    summary: Optional[str] = None
    debit_account: Optional[str] = None
    credit_account: Optional[str] = None
    debit_subaccount: Optional[str] = None
    credit_subaccount: Optional[str] = None
    invoice_status: Optional[str] = None


@app.post("/documents/{document_id}/ok")
def api_ok_document(document_id: int, payload: ConfirmIn):
    with get_session() as s:
        out_path = confirm_document(
            s,
            document_id=document_id,
            date=payload.date,
            amount=payload.amount,
            summary=payload.summary,
            debit=payload.debit_account,
            credit=payload.credit_account,
            debit_sub=payload.debit_subaccount,
            credit_sub=payload.credit_subaccount,
            invoice_status=payload.invoice_status,
        )
        return {"status": "ok", "csv": str(out_path)}


@app.post("/documents/{document_id}/check")
def api_check_document(document_id: int):
    with get_session() as s:
        mark_check_later(s, document_id)
        return {"status": DocumentStatusEnum.CHECK_LATER.value}


@app.delete("/documents/{document_id}")
def api_delete_document(document_id: int):
    with get_session() as s:
        delete_document(s, document_id)
        return {"status": "deleted"}


@app.post("/export")
def api_export(company_name: str, encoding: str = "utf-8", bom: bool = False, target_dir: str | None = None):
    with get_session() as s:
        company = ensure_company(s, company_name)
        try:
            out_path = export_confirmed_csv(
                s,
                company.id,
                encoding=encoding,
                bom=bom,
                output_dir=Path(target_dir) if target_dir else None,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"status": "ok", "csv": str(out_path)}


class CompanySettingsOut(BaseModel):
    default_output_dir: Optional[str] = None
    archive_base_dir: Optional[str] = None


@app.get("/settings", response_model=CompanySettingsOut)
def api_get_settings(company_name: str):
    with get_session() as s:
        company = ensure_company(s, company_name)
        cs = get_company_settings(s, company.id)
        return CompanySettingsOut(default_output_dir=cs.default_output_dir, archive_base_dir=cs.archive_base_dir)


class CompanySettingsIn(BaseModel):
    company_name: str
    default_output_dir: Optional[str] = None
    archive_base_dir: Optional[str] = None


@app.post("/settings", response_model=CompanySettingsOut)
def api_set_settings(payload: CompanySettingsIn):
    with get_session() as s:
        company = ensure_company(s, payload.company_name)
        cs = update_company_settings(s, company.id, payload.default_output_dir, payload.archive_base_dir)
        return CompanySettingsOut(default_output_dir=cs.default_output_dir, archive_base_dir=cs.archive_base_dir)


# Account settings (per-account: subaccounts and summary options)
class AccountSettingIn(BaseModel):
    company_name: str
    account_name: str
    subaccounts: list[str] | None = None
    summaries: list[str] | None = None


class AccountSettingOut(BaseModel):
    account_name: str
    subaccounts: list[str]
    summaries: list[str]


@app.get("/account_settings", response_model=list[AccountSettingOut])
def api_get_account_settings(company_name: str):
    import json as _json

    with get_session() as s:
        company = ensure_company(s, company_name)
        rows = get_account_settings(s, company.id)
        out: list[AccountSettingOut] = []
        for r in rows:
            subs = []
            sums = []
            try:
                subs = list(_json.loads(r.subaccounts_json or "[]"))
            except Exception:
                pass
            try:
                sums = list(_json.loads(r.summaries_json or "[]"))
            except Exception:
                pass
            out.append(AccountSettingOut(account_name=r.account_name, subaccounts=subs, summaries=sums))
        return out


@app.post("/account_settings", response_model=AccountSettingOut)
def api_set_account_setting(payload: AccountSettingIn):
    with get_session() as s:
        company = ensure_company(s, payload.company_name)
        row = upsert_account_setting(s, company.id, payload.account_name, payload.subaccounts, payload.summaries)
        import json as _json

        subs = list(_json.loads(row.subaccounts_json or "[]")) if row.subaccounts_json else []
        sums = list(_json.loads(row.summaries_json or "[]")) if row.summaries_json else []
        return AccountSettingOut(account_name=row.account_name, subaccounts=subs, summaries=sums)


# Admin: OCR master status and settings
class OCRStatusOut(BaseModel):
    samples: int
    last_added: Optional[str]
    active_lang: str
    model_dir: Optional[str]


@app.get("/admin/ocr_status", response_model=OCRStatusOut)
def api_ocr_status():
    with get_session() as s:
        samples = s.scalar(select(func.count(OCRMasterSample.id))) or 0
        last = s.scalar(select(func.max(OCRMasterSample.created_at)))
        settings = s.scalar(select(OCRMasterSettings).limit(1))
        if not settings:
            settings = OCRMasterSettings(active_lang="japan", model_dir=None)
            s.add(settings)
            s.flush()
        return OCRStatusOut(
            samples=samples,
            last_added=last.isoformat() if last else None,
            active_lang=settings.active_lang,
            model_dir=settings.model_dir,
        )


class OCRSettingsIn(BaseModel):
    active_lang: Optional[str] = None
    model_dir: Optional[str] = None


@app.post("/admin/ocr_settings", response_model=OCRStatusOut)
def api_set_ocr_settings(payload: OCRSettingsIn):
    with get_session() as s:
        settings = s.scalar(select(OCRMasterSettings).limit(1))
        if not settings:
            settings = OCRMasterSettings(active_lang=payload.active_lang or "japan", model_dir=payload.model_dir)
            s.add(settings)
        else:
            if payload.active_lang is not None:
                settings.active_lang = payload.active_lang
            if payload.model_dir is not None:
                settings.model_dir = payload.model_dir
        s.flush()
        samples = s.scalar(select(func.count(OCRMasterSample.id))) or 0
        last = s.scalar(select(func.max(OCRMasterSample.created_at)))
        return OCRStatusOut(
            samples=samples,
            last_added=last.isoformat() if last else None,
            active_lang=settings.active_lang,
            model_dir=settings.model_dir,
        )


# Admin: LLM settings (stored simply in-memory/env-like via settings table placeholder)
class LLMSettings(BaseModel):
    provider: str = "llama-cpp"
    model_path: Optional[str] = None  # GGUF file path
    device: str = "cpu"  # cpu or gpu
    n_gpu_layers: int = 0
    n_threads: int = 4


_LLM_SETTINGS: LLMSettings = LLMSettings()


@app.get("/admin/llm_settings", response_model=LLMSettings)
def api_get_llm_settings():
    return _LLM_SETTINGS


@app.post("/admin/llm_settings", response_model=LLMSettings)
def api_set_llm_settings(payload: LLMSettings):
    global _LLM_SETTINGS
    _LLM_SETTINGS = payload
    # Apply to runtime LLM client
    cfg = LLMConfig(
        provider=_LLM_SETTINGS.provider,
        model_path=_LLM_SETTINGS.model_path,
        device=_LLM_SETTINGS.device,
        n_gpu_layers=_LLM_SETTINGS.n_gpu_layers,
        n_threads=_LLM_SETTINGS.n_threads,
    )
    set_llm_config(cfg)
    return _LLM_SETTINGS


# Company-level LLM override settings
from ..models import CompanyLLMSetting, LLMLog


class CompanyLLMSettingsIn(BaseModel):
    company_name: str
    use_override: bool = False
    provider: str = "llama-cpp"
    model_path: Optional[str] = None
    device: str = "cpu"
    n_gpu_layers: int = 0
    n_threads: int = 4
    lora_path: Optional[str] = None
    prompt_template: Optional[str] = None


class CompanyLLMSettingsOut(BaseModel):
    use_override: bool
    provider: str
    model_path: Optional[str]
    device: str
    n_gpu_layers: int
    n_threads: int
    lora_path: Optional[str]
    prompt_template: Optional[str]


@app.get("/company_llm_settings", response_model=CompanyLLMSettingsOut)
def api_get_company_llm_settings(company_name: str):
    with get_session() as s:
        company = ensure_company(s, company_name)
        cs = s.scalar(select(CompanyLLMSetting).where(CompanyLLMSetting.company_id == company.id))
        if not cs:
            cs = CompanyLLMSetting(company_id=company.id)
            s.add(cs)
            s.flush()
        return CompanyLLMSettingsOut(
            use_override=bool(cs.use_override),
            provider=cs.provider,
            model_path=cs.model_path,
            device=cs.device,
            n_gpu_layers=cs.n_gpu_layers,
            n_threads=cs.n_threads,
            lora_path=cs.lora_path,
            prompt_template=cs.prompt_template,
        )


@app.post("/company_llm_settings", response_model=CompanyLLMSettingsOut)
def api_set_company_llm_settings(payload: CompanyLLMSettingsIn):
    with get_session() as s:
        company = ensure_company(s, payload.company_name)
        cs = s.scalar(select(CompanyLLMSetting).where(CompanyLLMSetting.company_id == company.id))
        if not cs:
            cs = CompanyLLMSetting(company_id=company.id)
            s.add(cs)
        cs.use_override = 1 if payload.use_override else 0
        cs.provider = payload.provider
        cs.model_path = payload.model_path
        cs.device = payload.device
        cs.n_gpu_layers = payload.n_gpu_layers
        cs.n_threads = payload.n_threads
        cs.lora_path = payload.lora_path
        cs.prompt_template = payload.prompt_template
        s.flush()
        return CompanyLLMSettingsOut(
            use_override=bool(cs.use_override),
            provider=cs.provider,
            model_path=cs.model_path,
            device=cs.device,
            n_gpu_layers=cs.n_gpu_layers,
            n_threads=cs.n_threads,
            lora_path=cs.lora_path,
            prompt_template=cs.prompt_template,
        )


class CompanyLLMLogOut(BaseModel):
    document_id: int
    model_id: Optional[str]
    device: Optional[str]
    confidence: Optional[float]
    created_at: str


@app.get("/company_llm_logs", response_model=list[CompanyLLMLogOut])
def api_get_company_llm_logs(company_name: str, limit: int = 50):
    with get_session() as s:
        company = ensure_company(s, company_name)
        rows = s.execute(
            select(LLMLog).where(LLMLog.company_id == company.id).order_by(LLMLog.created_at.desc()).limit(limit)
        ).scalars()
        out: list[CompanyLLMLogOut] = []
        for r in rows:
            out.append(
                CompanyLLMLogOut(
                    document_id=r.document_id,
                    model_id=r.model_id,
                    device=r.device,
                    confidence=r.confidence,
                    created_at=r.created_at.isoformat(),
                )
            )
        return out
