"""Utilities for exporting journal entries to the Yayoi Accounting import format.

The module defines the following high level abstractions:

* :class:`AccountSide` – represents debit or credit side attributes of a journal line.
* :class:`JournalEntry` – container for a single Yayoi journal line (1行).
* :class:`JournalExporter` – orchestrates validation, transformation, and file output.

The exporter enforces the field length and formatting rules described in the
prompt, ensuring the generated CSV can be imported by Yayoi Accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import csv
import io
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

LOGGER = logging.getLogger(__name__)


class JournalExportError(Exception):
    """Raised when journal data cannot be exported due to validation issues."""


# Maximum lengths in half-width characters for each field
MAX_LENGTHS = {
    "identifier_flag": 4,
    "voucher_number": 6,
    "closing_classification": 4,
    "transaction_date": 10,
    "account": 24,
    "sub_account": 24,
    "department": 24,
    "tax_category": 32,
    "amount": 11,
    "tax_amount": 11,
    "summary": 64,
    "reference_number": 10,
    "due_date": 10,
    "type_code": 1,
    "origin_code": 4,
    "memo": 180,
    "tag1": 3,
    "tag2": 1,
}


def _truncate(value: str, max_length: int) -> str:
    """Truncate value to a maximum length, preserving half-width characters.

    Parameters
    ----------
    value:
        Input string value (None will be treated as an empty string).
    max_length:
        Maximum number of characters (half-width) permitted by the Yayoi spec.
    """

    value = value or ""
    if len(value) <= max_length:
        return value
    LOGGER.debug("Truncating value '%s' to %d characters", value, max_length)
    return value[:max_length]


def _validate_numeric(value: Optional[int], field_name: str) -> int:
    """Ensure numeric fields are integers and non-null."""

    if value is None:
        raise JournalExportError(f"Field '{field_name}' is required and must be an integer.")
    if isinstance(value, bool):  # Guard against booleans (subclass of int)
        raise JournalExportError(f"Field '{field_name}' cannot be a boolean.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise JournalExportError(
            f"Field '{field_name}' must be convertible to int. Got {value!r}."
        ) from exc


def _format_date(value: Optional[str], field_name: str) -> str:
    """Validate and normalise date strings.

    Yayoi accepts several formats. This helper ensures common ISO dates are
    converted to ``YYYY/MM/DD`` while leaving custom era-based strings as-is.
    """

    if not value:
        return ""

    value = value.strip()

    # Allow already compliant era or slash separated formats
    if "-" in value or "/" in value:
        try:
            parsed = datetime.strptime(value.replace("-", "/"), "%Y/%m/%d")
            return parsed.strftime("%Y/%m/%d")
        except ValueError:
            # Assume custom format (e.g. Reiwa era "R01/07/01")
            return value

    # Accept compact YYYYMMDD format
    if len(value) == 8 and value.isdigit():
        try:
            parsed = datetime.strptime(value, "%Y%m%d")
            return parsed.strftime("%Y/%m/%d")
        except ValueError as exc:
            raise JournalExportError(
                f"Field '{field_name}' must be a valid date. Got {value!r}."
            ) from exc

    # Fallback: raise an error for unknown patterns
    raise JournalExportError(f"Field '{field_name}' has invalid date format: {value!r}")


def _format_boolean(value: Optional[object]) -> str:
    """Convert truthy values to the accepted Yayoi adjustment flags."""

    truthy = {"yes", "true", "on", "1", "-1", True, 1, -1}
    falsy = {"no", "false", "off", "0", False, 0, None, ""}

    if value in truthy:
        return "yes"
    if value in falsy:
        return "no"

    # Default to "no" but log for traceability
    LOGGER.debug("Unrecognised boolean flag %r treated as 'no'", value)
    return "no"


@dataclass
class AccountSide:
    """Represents the debit or credit side of a journal line."""

    account: str
    sub_account: str = ""
    department: str = ""
    tax_category: str = ""
    amount: int = 0
    tax_amount: int = 0

    def sanitise(self, prefix: str) -> List[str]:
        """Validate and format the account side for CSV output."""

        formatted_amount = f"{_validate_numeric(self.amount, prefix + 'amount'):d}"
        formatted_tax_amount = f"{_validate_numeric(self.tax_amount, prefix + 'tax_amount'):d}"

        return [
            _truncate(self.account, MAX_LENGTHS["account"]),
            _truncate(self.sub_account, MAX_LENGTHS["sub_account"]),
            _truncate(self.department, MAX_LENGTHS["department"]),
            _truncate(self.tax_category, MAX_LENGTHS["tax_category"]),
            formatted_amount,
            formatted_tax_amount,
        ]


@dataclass
class JournalEntry:
    """Represents a single line of a Yayoi journal voucher."""

    identifier_flag: str
    voucher_number: str
    transaction_date: str
    debit: AccountSide
    credit: AccountSide
    summary: str = ""
    reference_number: str = ""
    due_date: str = ""
    type_code: str = "0"
    origin_code: str = ""
    memo: str = ""
    tag1: str = ""
    tag2: str = ""
    adjustment_flag: Optional[object] = None
    closing_classification: str = ""

    def to_csv_row(self) -> List[str]:
        """Convert the entry to a Yayoi-compliant CSV row."""

        identifier = _truncate(self.identifier_flag, MAX_LENGTHS["identifier_flag"])
        voucher_no = _truncate(self.voucher_number, MAX_LENGTHS["voucher_number"])
        closing = _truncate(self.closing_classification, MAX_LENGTHS["closing_classification"])
        date = _truncate(_format_date(self.transaction_date, "transaction_date"), MAX_LENGTHS["transaction_date"])

        debit_fields = self.debit.sanitise("debit_")
        credit_fields = self.credit.sanitise("credit_")

        summary = _truncate(self.summary, MAX_LENGTHS["summary"])
        ref = _truncate(self.reference_number, MAX_LENGTHS["reference_number"])
        due = _truncate(_format_date(self.due_date, "due_date") if self.due_date else "", MAX_LENGTHS["due_date"])
        type_code = _truncate(self.type_code or "0", MAX_LENGTHS["type_code"])
        origin = _truncate(self.origin_code, MAX_LENGTHS["origin_code"])
        memo = _truncate(self.memo, MAX_LENGTHS["memo"])
        tag1 = _truncate(self.tag1, MAX_LENGTHS["tag1"])
        tag2 = _truncate(self.tag2, MAX_LENGTHS["tag2"])
        adjustment = _format_boolean(self.adjustment_flag)

        row = [
            identifier,
            voucher_no,
            closing,
            date,
            *debit_fields,
            *credit_fields,
            summary,
            ref,
            due,
            type_code,
            origin,
            memo,
            tag1,
            tag2,
            adjustment,
        ]

        return row


@dataclass
class JournalExporter:
    """Handles conversion from journal entries to Yayoi-compliant CSV."""

    entries: Sequence[JournalEntry] = field(default_factory=list)

    def to_csv_string(self) -> str:
        """Generate the CSV content as a string."""

        with io.StringIO(newline="") as buffer:
            writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            for entry in self.entries:
                writer.writerow(entry.to_csv_row())
            return buffer.getvalue()

    def save(self, output_path: Path | str, encoding: str = "utf-8", bom: bool = False) -> Path:
        """Persist the CSV to disk.

        Parameters
        ----------
        output_path:
            Target path for the generated file.
        encoding:
            Output encoding (default UTF-8). Use "utf-8-sig" to emit BOM.
        bom:
            If ``True`` and ``encoding`` is ``"utf-8"``, a BOM will be prepended.
        """

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        csv_content = self.to_csv_string()

        if bom and encoding.lower().replace("_", "-") == "utf-8":
            csv_bytes = csv_content.encode("utf-8")
            with path.open("wb") as fh:
                fh.write(b"\xef\xbb\xbf" + csv_bytes)
        else:
            with path.open("w", encoding=encoding, newline="") as fh:
                fh.write(csv_content)

        LOGGER.info("Exported %d journal entries to %s", len(self.entries), path)
        return path

    @classmethod
    def from_dicts(cls, payload: Iterable[dict]) -> "JournalExporter":
        """Create an exporter from an iterable of dictionaries.

        Each dictionary must contain at least the minimum required keys for a
        :class:`JournalEntry`. Missing mandatory fields raise
        :class:`JournalExportError` with descriptive messages.
        """

        entries: List[JournalEntry] = []
        for raw in payload:
            try:
                entry = JournalEntry(
                    identifier_flag=raw["identifier_flag"],
                    voucher_number=str(raw["voucher_number"]),
                    transaction_date=raw["transaction_date"],
                    debit=_build_account_side(raw.get("debit", {}), "debit"),
                    credit=_build_account_side(raw.get("credit", {}), "credit"),
                    summary=raw.get("summary", ""),
                    reference_number=raw.get("reference_number", ""),
                    due_date=raw.get("due_date", ""),
                    type_code=str(raw.get("type_code", "0")),
                    origin_code=raw.get("origin_code", ""),
                    memo=raw.get("memo", ""),
                    tag1=str(raw.get("tag1", "")),
                    tag2=str(raw.get("tag2", "")),
                    adjustment_flag=raw.get("adjustment_flag"),
                    closing_classification=raw.get("closing_classification", ""),
                )
            except KeyError as exc:
                raise JournalExportError(f"Missing required field: {exc.args[0]}") from exc

            entries.append(entry)

        return cls(entries=entries)


def _build_account_side(data: dict, prefix: str) -> AccountSide:
    """Create an :class:`AccountSide` from a dictionary."""

    try:
        account = data["account"]
    except KeyError as exc:
        raise JournalExportError(f"Missing required field: {prefix}.account") from exc

    amount = data.get("amount")
    tax_amount = data.get("tax_amount", 0)

    return AccountSide(
        account=account,
        sub_account=data.get("sub_account", ""),
        department=data.get("department", ""),
        tax_category=data.get("tax_category", ""),
        amount=_validate_numeric(amount, f"{prefix}.amount"),
        tax_amount=_validate_numeric(tax_amount, f"{prefix}.tax_amount"),
    )


__all__ = [
    "AccountSide",
    "JournalEntry",
    "JournalExporter",
    "JournalExportError",
]
