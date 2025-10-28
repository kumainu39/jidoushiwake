from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import requests
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QDialog,
)

API_URL = "http://127.0.0.1:8765"


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

        layout.addWidget(existing_group)
        layout.addWidget(new_group)
        self.setLayout(layout)

        self.selected: Optional[str] = None
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
        self.list_widget = QListWidget()
        layout.addWidget(QLabel("未確認一覧"))
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
        layout = QVBoxLayout()
        layout.addWidget(QLabel("仕訳確認（未確認一覧）"))
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        self.list_widget.clear()
        try:
            r = requests.get(f"{API_URL}/documents", params={"company_name": self.company, "status": "unconfirmed"}, timeout=10)
            for item in r.json() or []:
                lw = QListWidgetItem(f"#{item['id']} - {Path(item['file_path']).name}")
                self.list_widget.addItem(lw)
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
        for p in (self.scan_page, self.review_page):
            self.stack.addWidget(p)
        self.setCentralWidget(self.stack)

        menubar = self.menuBar()
        act_company = menubar.addAction("会社")
        act_scan = menubar.addAction("スキャン")
        act_review = menubar.addAction("仕訳確認")
        act_company.triggered.connect(self.change_company)  # type: ignore[arg-type]
        act_scan.triggered.connect(lambda: self.stack.setCurrentWidget(self.scan_page))  # type: ignore[arg-type]
        act_review.triggered.connect(lambda: self.stack.setCurrentWidget(self.review_page))  # type: ignore[arg-type]

    def change_company(self) -> None:
        dlg = CompanySelector()
        res = dlg.exec()
        if res == QDialog.DialogCode.Accepted and dlg.selected:
            self.__init__(dlg.selected)  # re-init window with new company


def run_ui() -> None:
    app = QApplication(sys.argv)
    selector = CompanySelector()
    result = selector.exec()
    if result != QDialog.DialogCode.Accepted or not selector.selected:
        return
    win = MainWindow(selector.selected)
    win.resize(1100, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_ui()
