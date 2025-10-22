import csv
import io
import pytest

from jidoushiwake.yayoi_exporter import (
    AccountSide,
    JournalEntry,
    JournalExporter,
    JournalExportError,
)


def test_to_csv_row_truncates_fields():
    entry = JournalEntry(
        identifier_flag="2000123",
        voucher_number="1234567",
        transaction_date="2024-01-31",
        debit=AccountSide(account="売上高" * 10, amount=1000, tax_amount=100),
        credit=AccountSide(account="現金" * 10, amount=1000, tax_amount=100),
        summary="摘要" * 40,
        reference_number="12345678901",
        memo="メモ" * 100,
        tag1="1234",
        tag2="12",
        adjustment_flag=True,
    )

    row = entry.to_csv_row()
    assert row[0] == "2000"  # identifier truncated to 4 chars
    assert row[1] == "123456"  # voucher number truncated to 6 digits
    assert len(row[16]) <= 64  # summary truncated to max length
    assert len(row[17]) <= 10  # reference truncated
    assert len(row[21]) <= 180
    assert row[22] == "123"  # tag1 truncated to 3 characters
    assert row[23] == "1"  # tag2 truncated to 1 character
    assert row[24] == "yes"  # truthy adjustment flag converted to yes


def test_exporter_to_csv_string():
    entries = [
        JournalEntry(
            identifier_flag="2000",
            voucher_number="1",
            transaction_date="20240131",
            debit=AccountSide(
                account="売掛金",
                sub_account="A社",
                tax_category="課税売上8%",
                amount=1200,
                tax_amount=96,
            ),
            credit=AccountSide(
                account="売上高",
                tax_category="課税売上8%",
                amount=1200,
                tax_amount=96,
            ),
            summary="テスト取引",
            reference_number="INV-001",
            due_date="2024-02-29",
            type_code="3",
            origin_code="販売",
            memo="スキャナ仕訳",
            tag1="001",
            tag2="1",
            adjustment_flag="no",
        )
    ]

    exporter = JournalExporter(entries)
    csv_content = exporter.to_csv_string()
    reader = csv.reader(io.StringIO(csv_content))
    rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row[3] == "2024/01/31"  # date formatted with slashes
    assert row[7] == "課税売上8%"  # debit tax category
    assert row[13] == "課税売上8%"  # credit tax category
    assert row[8] == "1200"
    assert row[14] == "1200"
    assert row[18] == "2024/02/29"
    assert row[19] == "3"
    assert row[20] == "販売"
    assert row[21] == "スキャナ仕訳"
    assert row[-1] == "no"


def test_from_dicts_validates_required_fields():
    with pytest.raises(JournalExportError):
        JournalExporter.from_dicts([
            {
                "identifier_flag": "2000",
                "voucher_number": "1",
                "transaction_date": "2024/01/01",
                "debit": {"amount": 100, "tax_amount": 10},  # missing account
                "credit": {"account": "現金", "amount": 100, "tax_amount": 10},
            }
        ])

    with pytest.raises(JournalExportError):
        JournalExporter.from_dicts([
            {
                "identifier_flag": "2000",
                "voucher_number": "1",
                "transaction_date": "2024/01/01",
                "debit": {"account": "現金", "amount": None, "tax_amount": 10},
                "credit": {"account": "売上高", "amount": 100, "tax_amount": 10},
            }
        ])
