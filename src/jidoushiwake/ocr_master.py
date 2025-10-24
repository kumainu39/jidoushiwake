from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Integer, String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class OCRMasterSample(Base):
    __tablename__ = "ocr_master_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pdf_path: Mapped[str] = mapped_column(Text)
    page_index: Mapped[int] = mapped_column(Integer, default=0)
    extracted_text: Mapped[str] = mapped_column(Text)  # current OCR output
    ground_truth: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class OCRMasterSettings(Base):
    __tablename__ = "ocr_master_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    active_lang: Mapped[str] = mapped_column(String(32), default="japan")
    model_dir: Mapped[Optional[str]] = mapped_column(Text)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

