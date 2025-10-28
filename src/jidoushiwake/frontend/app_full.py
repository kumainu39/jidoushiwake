from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import requests
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDialog,
)

API_URL = "http://127.0.0.1:8765"


def _render_pdf_first_page(path: Path, max_w: int = 600) -> Optional[QPixmap]:
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(path))
        if doc.page_count == 0:
            return None
        page = doc[0]
        zoom = 2.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(img)
        if pm.width() > max_w:
            pm = pm.scaledToWidth(max_w, Qt.TransformationMode.SmoothTransformation)
        return pm
    except Exception:
        return None


class CompanySelector(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("会社選択 / 新規作成")
        layout = QVBoxLayout()

        existing_group = QGroupBox("既存の会社")
        eg_layout = QHBoxLayout()
        self.combo = QComboBox()
        btn_refresh = QPushButton("更新")
        btn_select = QPushButton("選択")
        btn_refresh.clicked.connect(self.load_companies)  # type: ignore[arg-type]
        btn_select.clicked.connect(self.on_select)  # type: ignore[arg-type]
        eg_layout.addWidget(self.combo)
        eg_layout.addWidget(btn_refresh)
        eg_layout.addWidget(btn_select)
        existing_group.setLayout(eg_layout)

        new_group = QGroupBox("新規作成")
        ng_layout = QHBoxLayout()
        self.company_input = QLineEdit()
        self.company_input.setPlaceholderText("新しい会社名を入力")
        btn_create = QPushButton("作成")
        btn_create.clicked.connect(self.on_create)  # type: ignore[arg-type]
        ng_layout.addWidget(self.company_input)
        ng_layout.addWidget(btn_create)
        new_group.setLayout(ng_layout)

        btn_admin = QPushButton("管理")
        btn_admin.clicked.connect(self.on_admin)  # type: ignore[arg-type]

        layout.addWidget(existing_group)
        layout.addWidget(new_group)
        layout.addWidget(btn_admin)
        self.setLayout(layout)

        self.selected: Optional[str] = None
        self.admin_requested: bool = False
        self.load_companies()

    def load_companies(self) -> None:
        self.combo.clear()
        try:
            r = requests.get(f"{API_URL}/companies", timeout=5)
            for c in r.json() or []:
                name = (c or {}).get("name")
                if name:
                    self.combo.addItem(str(name))
        except Exception:
            pass

    def on_select(self) -> None:
        name = (self.combo.currentText() or "").strip()
        if not name:
            QMessageBox.warning(self, "選択エラー", "会社を選択してください")
            return
        self.selected = name
        self.accept()

    def on_create(self) -> None:
        name = (self.company_input.text() or "").strip()
        if not name:
            QMessageBox.warning(self, "入力エラー", "会社名を入力してください")
            return
        try:
            requests.post(f"{API_URL}/companies", json={"name": name}, timeout=5)
            self.selected = name
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"会社作成に失敗しました: {e}")

    def on_admin(self) -> None:
        self.admin_requested = True
        self.accept()


class ScanPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        layout.addWidget(QLabel("PDF/画像の取り込み"))
        row = QHBoxLayout()
        btn_files = QPushButton("ファイル取込")
        btn_files.clicked.connect(self.import_files)  # type: ignore[arg-type]
        btn_dir = QPushButton("フォルダ取込")
        btn_dir.clicked.connect(self.import_folder)  # type: ignore[arg-type]
        row.addWidget(btn_files)
        row.addWidget(btn_dir)
        layout.addLayout(row)
        self.list = QListWidget()
        layout.addWidget(QLabel("未確認一覧"))
        layout.addWidget(self.list)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        try:
            r = requests.get(f"{API_URL}/documents", params={"company_name": self.company, "status": "unconfirmed"}, timeout=10)
            for item in r.json() or []:
                t = f"#{item['id']} - {Path(item['file_path']).name}"
                lw = QListWidgetItem(t)
                lw.setData(Qt.ItemDataRole.UserRole, item)
                self.list.addItem(lw)
        except Exception:
            pass

    def _post_file(self, path: Path) -> bool:
        ext = path.suffix.lower()
        mime = (
            "application/pdf" if ext == ".pdf"
            else "image/jpeg" if ext in (".jpg", ".jpeg")
            else "image/png" if ext == ".png"
            else "image/tiff" if ext in (".tif", ".tiff")
            else "application/octet-stream"
        )
        try:
            with open(path, "rb") as fh:
                files_ = {"file": (path.name, fh, mime)}
                data = {"company_name": self.company}
                r = requests.post(f"{API_URL}/documents/import", data=data, files=files_, timeout=120)
                return bool(r.ok)
        except Exception:
            return False

    def import_files(self) -> None:
        filters = "PDF/画像 (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"
        files, _ = QFileDialog.getOpenFileNames(self, "ファイルを選択", str(Path.cwd()), filters)
        ok = False
        for f in files:
            if self._post_file(Path(f)):
                ok = True
        self.refresh()
        if ok:
            try:
                win = self.window()
                if hasattr(win, 'review_page') and getattr(win, 'review_page') is not None:
                    win.review_page.refresh()  # type: ignore[attr-defined]
                    if hasattr(win, 'stack'):
                        win.stack.setCurrentWidget(win.review_page)  # type: ignore[attr-defined]
            except Exception:
                pass

    def import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "フォルダを選択", str(Path.cwd()))
        if not folder:
            return
        paths = [p for p in Path(folder).rglob('*') if p.suffix.lower() in {'.pdf','.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}]
        ok = False
        for p in paths:
            if self._post_file(p):
                ok = True
        self.refresh()
        if ok:
            try:
                win = self.window()
                if hasattr(win, 'review_page') and getattr(win, 'review_page') is not None:
                    win.review_page.refresh()  # type: ignore[attr-defined]
                    if hasattr(win, 'stack'):
                        win.stack.setCurrentWidget(win.review_page)  # type: ignore[attr-defined]
            except Exception:
                pass


class ReviewPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        self.current_id: Optional[int] = None

        root = QHBoxLayout()
        # left: list
        self.list = QListWidget()
        self.list.itemSelectionChanged.connect(self.on_select)  # type: ignore[arg-type]
        # center: preview
        center = QVBoxLayout()
        self.preview = QLabel("PDFプレビュー")
        self.preview.setMinimumSize(QSize(400, 500))
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.preview)
        # right: form
        form = QFormLayout()
        self.date = QLineEdit()
        self.amount = QLineEdit()
        self.summary = QLineEdit()
        self.debit = QLineEdit()
        self.debit_sub = QLineEdit()
        self.credit = QLineEdit()
        self.credit_sub = QLineEdit()
        self.counterparty = QLineEdit()
        self.invoice = QComboBox(); self.invoice.addItems(["", "適格", "非適格", "非課税"])  # simple options
        form.addRow("日付(YYYY/MM/DD)", self.date)
        form.addRow("金額", self.amount)
        form.addRow("摘要", self.summary)
        form.addRow("借方科目", self.debit)
        form.addRow("借方補助", self.debit_sub)
        form.addRow("貸方科目", self.credit)
        form.addRow("貸方補助", self.credit_sub)
        form.addRow("相手先", self.counterparty)
        form.addRow("請求書区分", self.invoice)
        btn_row = QHBoxLayout()
        self.btn_ok = QPushButton("保存/OK(学習)")
        self.btn_check = QPushButton("後で確認")
        self.btn_del = QPushButton("削除")
        self.btn_ok.clicked.connect(self.on_ok)  # type: ignore[arg-type]
        self.btn_check.clicked.connect(self.on_check)  # type: ignore[arg-type]
        self.btn_del.clicked.connect(self.on_delete)  # type: ignore[arg-type]
        btn_row.addWidget(self.btn_ok); btn_row.addWidget(self.btn_check); btn_row.addWidget(self.btn_del)

        right = QVBoxLayout(); right.addLayout(form); right.addLayout(btn_row); right.addStretch(1)

        splitter = QSplitter();
        leftw = QWidget(); lw = QVBoxLayout(); lw.addWidget(QLabel("未確認一覧")); lw.addWidget(self.list); leftw.setLayout(lw)
        midw = QWidget(); midw.setLayout(center)
        rightw = QWidget(); rightw.setLayout(right)
        splitter.addWidget(leftw); splitter.addWidget(midw); splitter.addWidget(rightw)
        splitter.setSizes([240, 520, 360])
        root.addWidget(splitter)
        self.setLayout(root)

        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        try:
            r = requests.get(f"{API_URL}/documents", params={"company_name": self.company, "status": "unconfirmed"}, timeout=10)
            for item in r.json() or []:
                t = f"#{item['id']} - {Path(item['file_path']).name}"
                lw = QListWidgetItem(t)
                lw.setData(Qt.ItemDataRole.UserRole, item)
                self.list.addItem(lw)
        except Exception:
            pass

    def on_select(self) -> None:
        items = self.list.selectedItems()
        if not items:
            return
        data = items[0].data(Qt.ItemDataRole.UserRole) or {}
        self.current_id = int(data.get("id"))
        auto = data.get("auto") or {}
        self.date.setText(auto.get("date") or "")
        self.amount.setText(str(auto.get("amount")) if auto.get("amount") is not None else "")
        self.summary.setText(auto.get("summary") or "")
        self.debit.setText(auto.get("debit_account") or "")
        self.debit_sub.setText(auto.get("debit_subaccount") or "")
        self.credit.setText(auto.get("credit_account") or "")
        self.credit_sub.setText(auto.get("credit_subaccount") or "")
        self.counterparty.setText(auto.get("counterparty") or "")
        inv = auto.get("invoice_status") or ""
        try:
            self.invoice.setCurrentText(inv)
        except Exception:
            pass
        # preview
        pm = _render_pdf_first_page(Path(data.get("file_path")))
        if pm:
            self.preview.setPixmap(pm)
        else:
            self.preview.setText("プレビュー不可")

    def _warn(self, title: str, msg: str) -> None:
        QMessageBox.warning(self, title, msg)

    def on_ok(self) -> None:
        if not self.current_id:
            return
        payload = {
            "date": (self.date.text() or None),
            "amount": int(self.amount.text()) if (self.amount.text() or "").strip().isdigit() else None,
            "summary": (self.summary.text() or None),
            "debit_account": (self.debit.text() or None),
            "credit_account": (self.credit.text() or None),
            "debit_subaccount": (self.debit_sub.text() or None),
            "credit_subaccount": (self.credit_sub.text() or None),
            "invoice_status": (self.invoice.currentText() or None),
        }
        try:
            r = requests.post(f"{API_URL}/documents/{self.current_id}/ok", json=payload, timeout=15)
            if r.ok:
                self.refresh()
                self.preview.setText("保存しました")
            else:
                self._warn("保存失敗", r.text)
        except Exception as e:
            self._warn("保存失敗", str(e))

    def on_check(self) -> None:
        if not self.current_id:
            return
        try:
            r = requests.post(f"{API_URL}/documents/{self.current_id}/check", timeout=8)
            if r.ok:
                self.refresh()
            else:
                self._warn("設定失敗", r.text)
        except Exception as e:
            self._warn("設定失敗", str(e))

    def on_delete(self) -> None:
        if not self.current_id:
            return
        try:
            r = requests.delete(f"{API_URL}/documents/{self.current_id}", timeout=8)
            if r.ok:
                self.refresh()
            else:
                self._warn("削除失敗", r.text)
        except Exception as e:
            self._warn("削除失敗", str(e))


class SettingsPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        grp = QGroupBox("出力設定")
        form = QFormLayout()
        self.default_dir = QLineEdit()
        self.archive_dir = QLineEdit()
        form.addRow("既定出力フォルダ", self.default_dir)
        form.addRow("アーカイブ基底", self.archive_dir)
        grp.setLayout(form)
        btn = QPushButton("保存")
        btn.clicked.connect(self.on_save)  # type: ignore[arg-type]
        layout.addWidget(grp); layout.addWidget(btn); layout.addStretch(1)
        self.setLayout(layout)
        self.load()

    def load(self) -> None:
        try:
            r = requests.get(f"{API_URL}/settings", params={"company_name": self.company}, timeout=10)
            if r.ok:
                d = r.json() or {}
                self.default_dir.setText(d.get("default_output_dir") or "")
                self.archive_dir.setText(d.get("archive_base_dir") or "")
        except Exception:
            pass

    def on_save(self) -> None:
        payload = {
            "company_name": self.company,
            "default_output_dir": (self.default_dir.text() or None),
            "archive_base_dir": (self.archive_dir.text() or None),
        }
        try:
            r = requests.post(f"{API_URL}/settings", json=payload, timeout=10)
            if r.ok:
                QMessageBox.information(self, "保存", "設定を保存しました")
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))


class ExportPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        form = QFormLayout()
        self.target_dir = QLineEdit()
        self.encoding = QComboBox(); self.encoding.addItems(["utf-8", "shift_jis"])
        self.bom = QComboBox(); self.bom.addItems(["なし", "あり"])
        btn_browse = QPushButton("参照")
        btn_browse.clicked.connect(self.choose_dir)  # type: ignore[arg-type]
        row = QHBoxLayout(); row.addWidget(self.target_dir); row.addWidget(btn_browse)
        form.addRow("出力先", row)
        form.addRow("文字コード", self.encoding)
        form.addRow("BOM", self.bom)
        btn = QPushButton("CSV出力")
        btn.clicked.connect(self.on_export)  # type: ignore[arg-type]
        layout.addLayout(form); layout.addWidget(btn); layout.addStretch(1)
        self.setLayout(layout)

    def choose_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "出力先フォルダ", str(Path.cwd()))
        if d:
            self.target_dir.setText(d)

    def on_export(self) -> None:
        payload = {
            "company_name": self.company,
            "encoding": self.encoding.currentText(),
            "bom": True if self.bom.currentText() == "あり" else False,
            "target_dir": (self.target_dir.text().strip() or None),
        }
        try:
            r = requests.post(f"{API_URL}/export", params=payload, timeout=15)
            if r.ok:
                QMessageBox.information(self, "出力", "CSVを書き出しました")
            else:
                QMessageBox.warning(self, "出力失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "出力失敗", str(e))


class DuplicatesPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        self.list = QListWidget()
        btn = QPushButton("重複候補を更新")
        btn.clicked.connect(self.refresh)  # type: ignore[arg-type]
        layout.addWidget(btn); layout.addWidget(self.list)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        try:
            r = requests.get(f"{API_URL}/duplicates", params={"company_name": self.company}, timeout=15)
            for g in r.json() or []:
                new_doc = g.get("new") or {}
                matches = g.get("matches") or []
                text = f"新規: #{new_doc.get('id')} vs {len(matches)} 件"
                lw = QListWidgetItem(text)
                lw.setData(Qt.ItemDataRole.UserRole, g)
                self.list.addItem(lw)
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        self.setWindowTitle(f"自動仕訳 - {company}")

        self.stack = QStackedWidget()
        self.scan_page = ScanPage(company)
        self.review_page = ReviewPage(company)
        self.dup_page = DuplicatesPage(company)
        self.settings_page = SettingsPage(company)
        self.export_page = ExportPage(company)
        for p in (self.scan_page, self.review_page, self.dup_page, self.settings_page, self.export_page):
            self.stack.addWidget(p)
        self.setCentralWidget(self.stack)

        menubar = self.menuBar()
        act_company = menubar.addAction("会社選択")
        act_scan = menubar.addAction("スキャン")
        act_review = menubar.addAction("仕訳確認")
        act_dup = menubar.addAction("重複候補")
        act_settings = menubar.addAction("設定")
        act_export = menubar.addAction("出力")

        act_company.triggered.connect(self.change_company)  # type: ignore[arg-type]
        act_scan.triggered.connect(lambda: self.stack.setCurrentWidget(self.scan_page))  # type: ignore[arg-type]
        act_review.triggered.connect(lambda: self.stack.setCurrentWidget(self.review_page))  # type: ignore[arg-type]
        act_dup.triggered.connect(lambda: self.stack.setCurrentWidget(self.dup_page))  # type: ignore[arg-type]
        act_settings.triggered.connect(lambda: self.stack.setCurrentWidget(self.settings_page))  # type: ignore[arg-type]
        act_export.triggered.connect(lambda: self.stack.setCurrentWidget(self.export_page))  # type: ignore[arg-type]

    def change_company(self) -> None:
        dlg = CompanySelector()
        res = dlg.exec()
        if dlg.admin_requested:
            try:
                from .admin import create_admin_window

                win = create_admin_window()
                win.show()
            except Exception:
                pass
            return
        if res == QDialog.DialogCode.Accepted and dlg.selected:
            self.__init__(dlg.selected)  # re-init


def run_ui() -> None:
    app = QApplication(sys.argv)
    selector = CompanySelector()
    result = selector.exec()
    if selector.admin_requested:
        try:
            from .admin import create_admin_window

            win = create_admin_window()
            win.show()
            sys.exit(app.exec())
        except Exception:
            return
    if result != QDialog.DialogCode.Accepted or not selector.selected:
        return
    win = MainWindow(selector.selected)
    win.resize(1200, 760)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_ui()

