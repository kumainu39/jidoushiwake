from __future__ import annotations

from datetime import datetime
from typing import Optional

from enum import Enum as PyEnum

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Float,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    documents: Mapped[list[Document]] = relationship(back_populates="company", cascade="all,delete-orphan")  # type: ignore[name-defined]
    keyword_mappings: Mapped[list[KeywordMapping]] = relationship(back_populates="company", cascade="all,delete-orphan")  # type: ignore[name-defined]
    settings: Mapped[Optional[CompanySetting]] = relationship(back_populates="company", uselist=False, cascade="all,delete-orphan")  # type: ignore[name-defined]


class DocumentStatusEnum(str, PyEnum):  # type: ignore[misc]
    UNCONFIRMED = "unconfirmed"
    CHECK_LATER = "check_later"
    CONFIRMED = "confirmed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    file_path: Mapped[str] = mapped_column(Text)
    ocr_text: Mapped[Optional[str]] = mapped_column(Text, default=None)
    status: Mapped[str] = mapped_column(String(32), default=DocumentStatusEnum.UNCONFIRMED.value, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    company: Mapped[Company] = relationship(back_populates="documents")
    auto_result: Mapped[Optional[AutoResult]] = relationship(back_populates="document", uselist=False, cascade="all,delete-orphan")  # type: ignore[name-defined]
    corrections: Mapped[list[Correction]] = relationship(back_populates="document", cascade="all,delete-orphan")  # type: ignore[name-defined]


class AutoResult(Base):
    __tablename__ = "auto_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), unique=True)
    date: Mapped[Optional[str]] = mapped_column(String(10))  # YYYY/MM/DD
    amount: Mapped[Optional[int]] = mapped_column(Integer)
    summary: Mapped[str] = mapped_column(String(128), default="")
    debit_account: Mapped[str] = mapped_column(String(64), default="")
    debit_subaccount: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    credit_account: Mapped[str] = mapped_column(String(64), default="")
    credit_subaccount: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    counterparty: Mapped[str] = mapped_column(String(128), default="")
    score: Mapped[int] = mapped_column(Integer, default=0)
    invoice_status: Mapped[Optional[str]] = mapped_column(String(16), default=None)  # 適格/非適格/非課税

    document: Mapped[Document] = relationship(back_populates="auto_result")


class CorrectionActionEnum(str, PyEnum):  # type: ignore[misc]
    OK = "ok"
    CHECK_LATER = "check_later"
    DELETE = "delete"


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    action: Mapped[str] = mapped_column(String(32))
    corrected_date: Mapped[Optional[str]] = mapped_column(String(10))
    corrected_amount: Mapped[Optional[int]] = mapped_column(Integer)
    corrected_summary: Mapped[Optional[str]] = mapped_column(String(128))
    debit_account: Mapped[Optional[str]] = mapped_column(String(64))
    credit_account: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped[Document] = relationship(back_populates="corrections")


class KeywordMapping(Base):
    __tablename__ = "keyword_mappings"
    __table_args__ = (UniqueConstraint("company_id", "keyword", name="uq_company_keyword"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    keyword: Mapped[str] = mapped_column(String(64))
    debit_account: Mapped[str] = mapped_column(String(64), default="")
    credit_account: Mapped[str] = mapped_column(String(64), default="")
    weight: Mapped[int] = mapped_column(Integer, default=1)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    company: Mapped[Company] = relationship(back_populates="keyword_mappings")


class CompanySetting(Base):
    __tablename__ = "company_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), unique=True, index=True)
    default_output_dir: Mapped[Optional[str]] = mapped_column(Text, default=None)
    archive_base_dir: Mapped[Optional[str]] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    company: Mapped[Company] = relationship(back_populates="settings")


class CompanyLLMSetting(Base):
    __tablename__ = "company_llm_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), unique=True, index=True)
    use_override: Mapped[bool] = mapped_column(Integer, default=0)  # 0=False, 1=True
    provider: Mapped[str] = mapped_column(String(32), default="llama-cpp")
    model_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    device: Mapped[str] = mapped_column(String(8), default="cpu")  # cpu/gpu
    n_gpu_layers: Mapped[int] = mapped_column(Integer, default=0)
    n_threads: Mapped[int] = mapped_column(Integer, default=4)
    lora_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    prompt_template: Mapped[Optional[str]] = mapped_column(Text, default=None)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AccountSetting(Base):
    __tablename__ = "account_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), index=True)
    account_name: Mapped[str] = mapped_column(String(64))
    # JSON-encoded arrays (comma-separated fallback allowed)
    subaccounts_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    summaries_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LLMLog(Base):
    __tablename__ = "llm_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    device: Mapped[Optional[str]] = mapped_column(String(16), default=None)
    prompt_excerpt: Mapped[Optional[str]] = mapped_column(Text, default=None)
    response_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    confidence: Mapped[Optional[float]] = mapped_column(Float, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
