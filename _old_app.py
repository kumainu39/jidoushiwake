from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import requests
from PyQt6.QtCore import Qt, QTimer, QEvent
from PyQt6.QtGui import QPixmap, QImage
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
    QScrollArea,
    QSizePolicy,
)
from .admin import create_admin_window

API_URL = "http://127.0.0.1:8765"


class CompanySelector(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("莨夂､ｾ驕ｸ謚・/ 譁ｰ隕丈ｽ懈・")
        layout = QVBoxLayout()

        # 譌｢蟄倥・莨夂､ｾ縺九ｉ驕ｸ謚・        existing_group = QGroupBox("譌｢蟄倥・莨夂､ｾ縺九ｉ驕ｸ謚・)
        eg_layout = QHBoxLayout()
        self.combo = QComboBox()
        btn_refresh = QPushButton("譖ｴ譁ｰ")
        btn_select = QPushButton("驕ｸ謚・)
        btn_refresh.clicked.connect(self.load_companies)  # type: ignore[arg-type]
        btn_select.clicked.connect(self.on_select)  # type: ignore[arg-type]
        eg_layout.addWidget(self.combo)
        eg_layout.addWidget(btn_refresh)
        eg_layout.addWidget(btn_select)
        existing_group.setLayout(eg_layout)

        # 譁ｰ隕丈ｽ懈・
        new_group = QGroupBox("譁ｰ隕丈ｽ懈・")
        ng_layout = QHBoxLayout()
        self.company_input = QLineEdit()
        self.company_input.setPlaceholderText("譁ｰ縺励＞莨夂､ｾ蜷阪ｒ蜈･蜉・)
        btn_create = QPushButton("菴懈・")
        btn_create.clicked.connect(self.on_create)  # type: ignore[arg-type]
        ng_layout.addWidget(self.company_input)
        ng_layout.addWidget(btn_create)
        new_group.setLayout(ng_layout)

        btn_admin = QPushButton("邂｡逅・)
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
            QMessageBox.warning(self, "驕ｸ謚槭お繝ｩ繝ｼ", "莨夂､ｾ繧帝∈謚槭＠縺ｦ縺上□縺輔＞")
            return
        self.selected = name
        self.accept()

    def on_create(self) -> None:
        name = (self.company_input.text() or "").strip()
        if not name:
            QMessageBox.warning(self, "蜈･蜉帙お繝ｩ繝ｼ", "莨夂､ｾ蜷阪ｒ蜈･蜉帙＠縺ｦ縺上□縺輔＞")
            return
        try:
            requests.post(f"{API_URL}/companies", json={"name": name}, timeout=5)
            self.selected = name
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "繧ｨ繝ｩ繝ｼ", f"莨夂､ｾ菴懈・縺ｫ螟ｱ謨・ {e}")

    def on_admin(self) -> None:
        self.admin_requested = True
        self.accept()


class OutputPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()

        dest_row = QHBoxLayout()
        self.dest_edit = QLineEdit()
        browse = QPushButton("蜿ら・")
        browse.clicked.connect(self.choose_dir)  # type: ignore[arg-type]
        dest_row.addWidget(QLabel("蜃ｺ蜉帙ヵ繧ｩ繝ｫ繝"))
        dest_row.addWidget(self.dest_edit)
        dest_row.addWidget(browse)

        self.info = QLabel()
        export_btn = QPushButton("CSV繧貞・蜉・)
        export_btn.clicked.connect(self.on_export)  # type: ignore[arg-type]

        self.list_widget = QListWidget()
        refresh_btn = QPushButton("螻･豁ｴ譖ｴ譁ｰ")
        refresh_btn.clicked.connect(self.refresh_history)  # type: ignore[arg-type]

        layout.addLayout(dest_row)
        layout.addWidget(self.info)
        layout.addWidget(export_btn)
        layout.addWidget(QLabel("驕主悉縺ｮ蜃ｺ蜉・))
        layout.addWidget(self.list_widget)
        layout.addWidget(refresh_btn)
        layout.addStretch(1)
        self.setLayout(layout)

        self.update_info()
        self.refresh_history()

    def update_info(self) -> None:
        base = Path(__file__).resolve().parents[3]
        default_dir = base / "output"
        try:
            r = requests.get(f"{API_URL}/settings", params={"company_name": self.company}, timeout=10)
            if r.ok:
                data = r.json()
                cd = data.get("default_output_dir") or str(default_dir)
                self.dest_edit.setText(cd)
                self.info.setText(f"蜃ｺ蜉帛・: {cd}")
            else:
                self.dest_edit.setText(str(default_dir))
                self.info.setText(f"蜃ｺ蜉帛・: {default_dir}")
        except Exception:
            self.dest_edit.setText(str(default_dir))
            self.info.setText(f"蜃ｺ蜉帛・: {default_dir}")

    def on_export(self) -> None:
        dest = (self.dest_edit.text() or "").strip()
        params = {"company_name": self.company}
        if dest:
            params["target_dir"] = dest
        try:
            r = requests.post(f"{API_URL}/export", params=params, timeout=60)
            if r.ok:
                path = (r.json() or {}).get("csv")
                QMessageBox.information(self, "蜃ｺ蜉・, f"蜃ｺ蜉帙＠縺ｾ縺励◆\n{path}")
                self.refresh_history()
            else:
                QMessageBox.warning(self, "蜃ｺ蜉帛､ｱ謨・, r.text)
        except Exception as e:
            QMessageBox.warning(self, "蜃ｺ蜉帛､ｱ謨・, str(e))

    def choose_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "蜃ｺ蜉帙ヵ繧ｩ繝ｫ繝驕ｸ謚・, self.dest_edit.text() or str(Path.home()))
        if d:
            self.dest_edit.setText(d)

    def _history_dir(self) -> Path:
        p = Path(self.dest_edit.text().strip())
        if p.exists() and p.is_dir():
            return p
        return Path(__file__).resolve().parents[3] / "output"

    def refresh_history(self) -> None:
        self.list_widget.clear()
        d = self._history_dir()
        try:
            for f in sorted(d.glob("*_yayoi.csv"), key=lambda x: x.stat().st_mtime, reverse=True)[:200]:
                self.list_widget.addItem(str(f))
        except Exception:
            pass


class ReviewPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company

        root = QHBoxLayout()

        # Left: PDF + NL
        left = QVBoxLayout()
        self.pdf_label = QLabel("PDF繝励Ξ繝薙Η繝ｼ")
        # 譌｢螳壹ｒ蟆代＠蠎・ａ縺ｫ遒ｺ菫晢ｼ亥・縺ｮ讓ｪ蟷・↓謌ｻ縺呻ｼ・        try:
            self.pdf_label.setMinimumWidth(420)
        except Exception:
            pass
        self.pdf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pdf_label.setStyleSheet("border:1px solid #ccc; background:white;")
        self.pdf_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.pdf_scroll = QScrollArea()
        self.pdf_scroll.setWidgetResizable(False)
        self.pdf_scroll.setWidget(self.pdf_label)
        left.addWidget(self.pdf_scroll, 3)

        nl = QHBoxLayout()
        # 閾ｪ辟ｶ險隱槭・蜈･蜉帙・謚倥ｊ霑斐＠蜿ｯ閭ｽ縺ｫ・磯聞譁・〒繧りｦ九ｄ縺吶￥・・        from PyQt6.QtWidgets import QTextEdit
        self.nl_edit = QTextEdit()
        try:
            self.nl_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        except Exception:
            pass
        self.nl_edit.setPlaceholderText("萓・ 繧ｹ繧ｿ繝・縺ｯ 莠､髫幄ｲｻ/迴ｾ驥・)
        try:
            self.nl_edit.setMinimumHeight(48)
            self.nl_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        except Exception:
            pass
        nl_btn = QPushButton("驕ｩ逕ｨ/蟄ｦ鄙・)
        nl_btn.clicked.connect(self._apply_nl)  # type: ignore[arg-type]
        nl.addWidget(self.nl_edit)
        nl.addWidget(nl_btn)
        left.addLayout(nl)

        # watch resizes
        self.pdf_label.installEventFilter(self)
        self.pdf_scroll.viewport().installEventFilter(self)

        # Right: grid + controls
        right = QVBoxLayout()
        toolbar = QHBoxLayout()
        self.btn_save = QPushButton("菫晏ｭ・OK(蟄ｦ鄙・")
        self.btn_later = QPushButton("蠕後〒遒ｺ隱・)
        self.btn_delete = QPushButton("蜑企勁")
        self.btn_save.clicked.connect(self._save_selected)  # type: ignore[arg-type]
        self.btn_later.clicked.connect(self._mark_later)    # type: ignore[arg-type]
        self.btn_delete.clicked.connect(self._delete_selected)  # type: ignore[arg-type]
        toolbar.addWidget(self.btn_save)
        toolbar.addWidget(self.btn_later)
        toolbar.addWidget(self.btn_delete)

        nav = QHBoxLayout()
        self.tax_combo = QComboBox()
        nav.addWidget(QLabel("遞主玄蛻・))
        nav.addWidget(self.tax_combo)
        tax_apply = QPushButton("遞主玄蛻・←逕ｨ")
        tax_apply.clicked.connect(self._apply_tax_combo)  # type: ignore[arg-type]
        nav.addWidget(tax_apply)
        nav.addStretch(1)
        self.pdf_prev = QPushButton("・憺・)
        self.pdf_next = QPushButton("鬆・ｼ・)
        self.pdf_page_label = QLabel("0/0")
        self.pdf_prev.clicked.connect(lambda: self._change_pdf_page(-1))  # type: ignore[arg-type]
        self.pdf_next.clicked.connect(lambda: self._change_pdf_page(1))   # type: ignore[arg-type]
        nav.addWidget(self.pdf_prev)
        nav.addWidget(self.pdf_page_label)
        nav.addWidget(self.pdf_next)
        self.doc_prev = QPushButton("・・)
        self.doc_next = QPushButton("・・)
        self.doc_label = QLabel("0/0")
        self.doc_prev.clicked.connect(lambda: self._change_doc(-1))  # type: ignore[arg-type]
        self.doc_next.clicked.connect(lambda: self._change_doc(1))   # type: ignore[arg-type]
        nav.addWidget(self.doc_prev)
        nav.addWidget(self.doc_label)
        nav.addWidget(self.doc_next)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["譌･莉・, "蛟滓婿", "蛟滓婿驥鷹｡・, "雋ｸ譁ｹ", "雋ｸ譁ｹ驥鷹｡・, "鞫倩ｦ・, "隲区ｱよ嶌蛹ｺ蛻・])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_table_select)  # type: ignore[arg-type]

        right.addLayout(toolbar)
        right.addLayout(nav)
        right.addWidget(self.table)

        lw = QWidget(); lw.setLayout(left)
        rw = QWidget(); rw.setLayout(right)
        spl = QSplitter(Qt.Orientation.Horizontal)
        spl.addWidget(lw)
        spl.addWidget(rw)
        # 蟾ｦ(PDF)蛛ｴ繧偵ｄ繧・ｺ・ａ縺ｫ・亥・縺ｮ豈皮紫縺ｫ謌ｻ縺呻ｼ・        spl.setStretchFactor(0, 2)
        spl.setStretchFactor(1, 3)
        root.addWidget(spl)
        self.setLayout(root)
        # 蛻晄悄繝ｬ繧､繧｢繧ｦ繝育｢ｺ螳壼ｾ後↓繧ｵ繧､繧ｺ豈斐ｒ霆ｽ縺剰ｪｿ謨ｴ
        try:
            QTimer.singleShot(0, lambda: spl.setSizes([560, 840]))
        except Exception:
            pass

        self._docs: list[dict] = []
        self._doc_index: int = 0
        self._pdf_path: Optional[str] = None
        self._pdf_page: int = 0

        self._load_tax()
        self.refresh()

    # ---------- Data loading ----------
    def refresh(self) -> None:
        try:
            r = requests.get(f"{API_URL}/documents", params={"company_name": self.company, "status": "unconfirmed"}, timeout=10)
            self._docs = list(r.json() or [])
        except Exception:
            self._docs = []
        if not self._docs:
            self.table.setRowCount(0)
            self._pdf_path = None
            self._render_labels()
            self._show_pdf(None)
            return
        if self._doc_index >= len(self._docs):
            self._doc_index = max(0, len(self._docs) - 1)
        self._render_current_doc()

    def _load_tax(self) -> None:
        try:
            r = requests.get(f"{API_URL}/tax_categories", timeout=10)
            self.tax_combo.clear()
            for row in r.json() or []:
                self.tax_combo.addItem(row.get("name") or "", row.get("code") or "")
        except Exception:
            pass

    # ---------- Rendering ----------
    def _render_labels(self) -> None:
        total = len(self._docs)
        self.doc_label.setText(f"{(self._doc_index+1) if total else 0}/{total}")

    def _render_current_doc(self) -> None:
        self._render_labels()
        d = self._docs[self._doc_index]
        auto = d.get("auto") or {}
        date = auto.get("date") or ""
        amount = abs(auto.get("amount") or 0)
        debit = auto.get("debit_account") or ""
        credit = auto.get("credit_account") or ""
        summary = auto.get("summary") or ""
        tax = auto.get("invoice_status") or ""

        self.table.setRowCount(2)
        # row0
        for j, v in enumerate([date, debit, f"{amount:,}", credit, f"{amount:,}", summary, tax]):
            self.table.setItem(0, j, QTableWidgetItem(str(v)))
        # row1 (subs/tax placeholder)
        for j, v in enumerate(["", "", "", "", "", tax, ""]):
            self.table.setItem(1, j, QTableWidgetItem(str(v)))

        self._pdf_page = 0
        self._show_pdf(d.get("file_path"))

    # ---------- Actions ----------
    def _selected_doc_id_and_rows(self) -> tuple[Optional[int], Optional[int], Optional[int]]:
        sel = self.table.selectedIndexes()
        # 繝・ヵ繧ｩ繝ｫ繝医〒迴ｾ蝨ｨPDF縺ｮ蜈磯ｭ陦後ｒ蟇ｾ雎｡縺ｫ縺吶ｋ・域悴驕ｸ謚槭〒繧ょ虚菴懶ｼ・        if not sel:
            if not self._docs:
                return None, None, None
            return int(self._docs[self._doc_index].get("id")), 0, 1
        r = sel[0].row()
        r0 = r - (r % 2)
        return int(self._docs[self._doc_index].get("id")), r0, r0 + 1

    def _collect_from_table(self, r0: int, r1: int) -> dict:
        def val(r, c):
            it = self.table.item(r, c)
            return it.text().strip() if it else ""
        date = val(r0, 0)
        debit = val(r0, 1)
        debit_amt = val(r0, 2)
        credit = val(r0, 3)
        credit_amt = val(r0, 4)
        summary = val(r0, 5)
        invoice = val(r0, 6)
        try:
            amount = int(debit_amt.replace(',', '')) if debit_amt else int(credit_amt.replace(',', ''))
        except Exception:
            amount = None
        return {
            "date": date or None,
            "amount": amount,
            "summary": summary or None,
            "debit": debit or None,
            "credit": credit or None,
            "invoice_status": invoice or None,
        }

    def _save_selected(self) -> None:
        doc_id, r0, r1 = self._selected_doc_id_and_rows()
        if doc_id is None or r0 is None or r1 is None:
            QMessageBox.information(self, "菫晏ｭ・, "陦後ｒ驕ｸ謚槭＠縺ｦ縺上□縺輔＞")
            return
        data = self._collect_from_table(r0, r1)
        try:
            requests.post(
                f"{API_URL}/documents/{doc_id}/ok",
                json={
                    "date": data["date"],
                    "amount": data["amount"],
                    "summary": data["summary"],
                    "debit_account": data["debit"],
                    "credit_account": data["credit"],
                    "invoice_status": data["invoice_status"],
                },
                timeout=20,
            )
            self.refresh()
        except Exception:
            pass

    def _mark_later(self) -> None:
        doc_id, _, _ = self._selected_doc_id_and_rows()
        if doc_id is None:
            return
        try:
            requests.post(f"{API_URL}/documents/{doc_id}/check", timeout=10)
            self.refresh()
        except Exception:
            pass

    def _delete_selected(self) -> None:
        doc_id, _, _ = self._selected_doc_id_and_rows()
        if doc_id is None:
            return
        # 遒ｺ隱阪ム繧､繧｢繝ｭ繧ｰ
        try:
            ret = QMessageBox.question(self, "蜑企勁遒ｺ隱・, f"繝峨く繝･繝｡繝ｳ繝・#{doc_id} 繧貞炎髯､縺励∪縺吶°・・)
            if ret != QMessageBox.StandardButton.Yes:
                return
        except Exception:
            pass
        try:
            r = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=10)
            if r.ok:
                self.refresh()
            else:
                QMessageBox.warning(self, "蜑企勁螟ｱ謨・, r.text)
        except Exception as e:
            QMessageBox.warning(self, "蜑企勁螟ｱ謨・, str(e))

    def _apply_nl(self) -> None:
        # QTextEdit/toPlainText 縺ｫ蟇ｾ蠢・        try:
            text = (self.nl_edit.toPlainText() or "").strip()
        except Exception:
            text = (getattr(self.nl_edit, 'text', lambda: '')() or '').strip()
        if not text:
            return
        try:
            requests.post(f"{API_URL}/company_nl_mapping", json={"company_name": self.company, "instruction": text}, timeout=15)
            QMessageBox.information(self, "蟄ｦ鄙・, "閾ｪ辟ｶ險隱槭・謖・､ｺ繧貞ｭｦ鄙偵＠縺ｾ縺励◆")
            try:
                self.nl_edit.clear()
            except Exception:
                self.nl_edit.setText("")
        except Exception:
            pass

    def _apply_tax_combo(self) -> None:
        sel = self.table.selectedIndexes()
        if not sel:
            return
        r = sel[0].row()
        r1 = r - (r % 2) + 1
        name = self.tax_combo.currentText()
        self.table.setItem(r1, 5, QTableWidgetItem(name))

    # ---------- Navigation ----------
    def _change_doc(self, d: int) -> None:
        if not self._docs:
            return
        self._doc_index = max(0, min(len(self._docs) - 1, self._doc_index + d))
        self._render_current_doc()

    def _change_pdf_page(self, d: int) -> None:
        if not self._pdf_path:
            return
        self._pdf_page = max(0, self._pdf_page + d)
        self._show_pdf(self._pdf_path)

    def _on_table_select(self) -> None:
        try:
            d = self._docs[self._doc_index]
            self._show_pdf(d.get("file_path"))
        except Exception:
            pass

    # ---------- PDF preview ----------
    def eventFilter(self, obj, event):  # type: ignore[override]
        if event.type() == QEvent.Type.Resize and (obj is self.pdf_label or obj is self.pdf_scroll.viewport()):
            QTimer.singleShot(0, lambda: self._show_pdf(self._pdf_path))
        return super().eventFilter(obj, event)

    def _show_pdf(self, path: Optional[str]) -> None:
        self._pdf_path = path or None
        if not path:
            self.pdf_label.setText("PDF縺ｪ縺・)
            self.pdf_page_label.setText("0/0")
            return
        try:
            import fitz  # PyMuPDF
        except Exception:
            self.pdf_label.setText("PyMuPDF譛ｪ繧､繝ｳ繧ｹ繝医・繝ｫ")
            return
        try:
            doc = fitz.open(path)
            if self._pdf_page >= len(doc):
                self._pdf_page = max(0, len(doc) - 1)
            page = doc.load_page(self._pdf_page)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            fmt = QImage.Format.Format_RGBA8888 if getattr(pix, 'alpha', False) else QImage.Format.Format_RGB888
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt).copy()
            pm = QPixmap.fromImage(img)
            try:
                rect = self.pdf_scroll.viewport().contentsRect()
                vpw = int(rect.width())
                vph = int(rect.height())
            except Exception:
                vpw = int(self.pdf_label.width())
                vph = 9999
            vpw = max(320, vpw - 4)
            vph = max(200, vph - 4)
            # 荳｡霎ｺ繝輔ぅ繝・ヨ・亥・菴楢｡ｨ遉ｺ・・            pm = pm.scaled(vpw, vph, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.pdf_label.setPixmap(pm)
            try:
                self.pdf_label.resize(pm.size())
            except Exception:
                pass
            self.pdf_page_label.setText(f"{self._pdf_page+1}/{len(doc)}")
            doc.close()
        except Exception as e:
            self.pdf_label.setText(f"PDF陦ｨ遉ｺ繧ｨ繝ｩ繝ｼ\n{e}")


class ScanPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        layout.addWidget(QLabel("PDF縺ｮ蜿悶ｊ霎ｼ縺ｿ"))
        btn = QPushButton("PDF蜿冶ｾｼ")
        btn.clicked.connect(self.import_pdfs)  # type: ignore[arg-type]
        self.list_widget = QListWidget()
        layout.addWidget(btn)
        layout.addWidget(QLabel("譛ｪ遒ｺ隱阪ョ繝ｼ繧ｿ"))
        layout.addWidget(self.list_widget)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        self.list_widget.clear()
        try:
            r = requests.get(f"{API_URL}/documents", params={"company_name": self.company, "status": "unconfirmed"}, timeout=10)
            for item in r.json() or []:
                lw = QListWidgetItem(f"#{item['id']} - {Path(item['file_path']).name}")
                lw.setData(Qt.ItemDataRole.UserRole, item)
                self.list_widget.addItem(lw)
        except Exception:
            pass

    def import_pdfs(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "PDF繧帝∈謚・, str(Path.cwd()), "PDF Files (*.pdf)")
        imported_any = False
        for f in files:
            try:
                with open(f, "rb") as fh:
                    files_ = {"file": (Path(f).name, fh, "application/pdf")}
                    data = {"company_name": self.company}
                    r = requests.post(f"{API_URL}/documents/import", data=data, files=files_, timeout=60)
                    if r.ok:
                        imported_any = True
            except Exception as e:
                QMessageBox.warning(self, "蜿冶ｾｼ螟ｱ謨・, f"{f}: {e}")
        # 閾ｪ繝壹・繧ｸ譖ｴ譁ｰ + 莉戊ｨｳ遒ｺ隱阪・繝ｼ繧ｸ繧ょ叉譎ょ渚譏
        self.refresh()
        try:
            win = self.window()
            if hasattr(win, 'review_page') and getattr(win, 'review_page') is not None:
                win.review_page.refresh()  # type: ignore[attr-defined]
                # 蜿悶ｊ霎ｼ縺ｿ縺ｫ謌仙粥縺励※縺・ｌ縺ｰ縲∽ｻ戊ｨｳ遒ｺ隱阪ち繝悶↓閾ｪ蜍輔〒驕ｷ遘ｻ
                if imported_any and hasattr(win, 'stack'):
                    win.stack.setCurrentWidget(win.review_page)  # type: ignore[attr-defined]
        except Exception:
            pass


class CheckPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout(); layout.addWidget(QLabel("隕√メ繧ｧ繝・け・育ｰ｡譏楢｡ｨ遉ｺ・・)); self.setLayout(layout)


class DuplicatesPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout(); layout.addWidget(QLabel("驥崎､・呵｣懶ｼ育ｰ｡譏楢｡ｨ遉ｺ・・)); self.setLayout(layout)


class SettingsPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        grp = QGroupBox("蜃ｺ蜉幄ｨｭ螳・)
        form = QFormLayout()
        base = Path(__file__).resolve().parents[3]
        self.default_label = QLabel(f"app/output ({base / 'output'})")
        self.custom_default_edit = QLineEdit()
        form.addRow("譌｢螳・app/output", self.default_label)
        form.addRow("繧ｫ繧ｹ繧ｿ繝譌｢螳壹ヱ繧ｹ", self.custom_default_edit)
        grp.setLayout(form)
        btn = QPushButton("菫晏ｭ・)
        btn.clicked.connect(self.on_save)  # type: ignore[arg-type]
        layout.addWidget(grp)
        layout.addWidget(btn)
        layout.addStretch(1)
        self.setLayout(layout)
        self.load_settings()

    def on_save(self) -> None:
        path = (self.custom_default_edit.text() or "").strip() or None
        try:
            r = requests.post(f"{API_URL}/settings", json={"company_name": self.company, "default_output_dir": path}, timeout=10)
            if r.ok:
                QMessageBox.information(self, "菫晏ｭ・, "險ｭ螳壹ｒ菫晏ｭ倥＠縺ｾ縺励◆")
            else:
                QMessageBox.warning(self, "菫晏ｭ伜､ｱ謨・, r.text)
        except Exception as e:
            QMessageBox.warning(self, "菫晏ｭ伜､ｱ謨・, str(e))

    def load_settings(self) -> None:
        try:
            r = requests.get(f"{API_URL}/settings", params={"company_name": self.company}, timeout=10)
            if r.ok:
                d = r.json() or {}
                self.custom_default_edit.setText(d.get("default_output_dir") or "")
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        self.setWindowTitle(f"閾ｪ蜍穂ｻ戊ｨｳ - {company}")

        self.stack = QStackedWidget()
        self.scan_page = ScanPage(company)
        self.review_page = ReviewPage(company)
        self.check_page = CheckPage(company)
        self.output_page = OutputPage(company)
        self.dup_page = DuplicatesPage(company)
        self.settings_page = SettingsPage(company)
        for p in (self.scan_page, self.review_page, self.check_page, self.output_page, self.dup_page, self.settings_page):
            self.stack.addWidget(p)
        self.setCentralWidget(self.stack)

        menubar = self.menuBar()
        act_company = menubar.addAction("莨夂､ｾ驕ｸ謚・)
        act_scan = menubar.addAction("繧ｹ繧ｭ繝｣繝ｳ")
        act_review = menubar.addAction("莉戊ｨｳ遒ｺ隱・)
        act_check = menubar.addAction("隕√メ繧ｧ繝・け")
        act_export = menubar.addAction("蜃ｺ蜉・)
        act_dup = menubar.addAction("驥崎､・､懃衍")
        act_settings = menubar.addAction("險ｭ螳・)

        act_company.triggered.connect(self.change_company)  # type: ignore[arg-type]
        act_scan.triggered.connect(lambda: self.stack.setCurrentWidget(self.scan_page))  # type: ignore[arg-type]
        act_review.triggered.connect(lambda: self.stack.setCurrentWidget(self.review_page))  # type: ignore[arg-type]
        act_check.triggered.connect(lambda: self.stack.setCurrentWidget(self.check_page))  # type: ignore[arg-type]
        act_export.triggered.connect(self.show_output)  # type: ignore[arg-type]
        act_dup.triggered.connect(lambda: self.stack.setCurrentWidget(self.dup_page))  # type: ignore[arg-type]
        act_settings.triggered.connect(lambda: self.stack.setCurrentWidget(self.settings_page))  # type: ignore[arg-type]

    def show_output(self) -> None:
        self.output_page.update_info()
        self.stack.setCurrentWidget(self.output_page)

    def change_company(self) -> None:
        dlg = CompanySelector()
        res = dlg.exec()
        if getattr(dlg, 'admin_requested', False):
            try:
                win = create_admin_window()
                win.show()
            except Exception:
                pass
            return
        if res == QDialog.DialogCode.Accepted and dlg.selected:
            self.__init__(dlg.selected)  # re-init window with new company


def run_ui() -> None:
    app = QApplication(sys.argv)
    selector = CompanySelector()
    result = selector.exec()
    if selector.admin_requested:
        try:
            admin_win = create_admin_window()
            admin_win.show()
            sys.exit(app.exec())
        except Exception:
            return
    if result != QDialog.DialogCode.Accepted or not selector.selected:
        return
    win = MainWindow(selector.selected)
    win.resize(1200, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_ui()
