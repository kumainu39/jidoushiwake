from __future__ import annotations

import sys
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QMainWindow,
    QStackedWidget,
    QListWidget,
    QDialog,
)
import requests
from PyQt6.QtWidgets import QListWidgetItem, QCheckBox, QTextEdit


API_URL = "http://127.0.0.1:8765"


class OCRStatusPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.samples_label = QLabel("サンプル数: -")
        self.last_label = QLabel("最終追加: -")
        form = QFormLayout()
        self.active_lang = QLineEdit()
        self.model_dir = QLineEdit()
        form.addRow("OCR言語", self.active_lang)
        form.addRow("モデルディレクトリ", self.model_dir)
        save = QPushButton("保存")
        save.clicked.connect(self.save)  # type: ignore[arg-type]
        layout.addWidget(self.samples_label)
        layout.addWidget(self.last_label)
        layout.addLayout(form)
        layout.addWidget(save)
        layout.addStretch(1)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        try:
            r = requests.get(f"{API_URL}/admin/ocr_status", timeout=10)
            if r.ok:
                data = r.json()
                self.samples_label.setText(f"サンプル数: {data.get('samples')}")
                self.last_label.setText(f"最終追加: {data.get('last_added')}")
                self.active_lang.setText(data.get("active_lang") or "")
                self.model_dir.setText(data.get("model_dir") or "")
        except Exception:
            pass

    def save(self) -> None:
        payload = {
            "active_lang": self.active_lang.text().strip() or None,
            "model_dir": self.model_dir.text().strip() or None,
        }
        try:
            r = requests.post(f"{API_URL}/admin/ocr_settings", json=payload, timeout=10)
            if r.ok:
                QMessageBox.information(self, "保存", "OCR設定を保存しました")
                self.refresh()
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))


class LLMSettingsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        form = QFormLayout()
        self.provider = QLineEdit("llama-cpp")
        self.model_path = QLineEdit("F:\\models\\Llama-3-ELYZA-JP-8B-q4_k_m.gguf")
        # GPUのみ前提（変更不可）
        self.device = QLineEdit("gpu")
        try:
            self.device.setReadOnly(True)
        except Exception:
            pass
        self.n_gpu_layers = QLineEdit("-1")
        self.n_threads = QLineEdit("4")
        form.addRow("Provider", self.provider)
        form.addRow("Model Path (GGUF)", self.model_path)
        form.addRow("Device (GPUのみ)", self.device)
        form.addRow("GPU Layers", self.n_gpu_layers)
        form.addRow("Threads", self.n_threads)
        save = QPushButton("保存")
        save.clicked.connect(self.save)  # type: ignore[arg-type]
        layout.addLayout(form)
        layout.addWidget(save)
        layout.addStretch(1)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        try:
            r = requests.get(f"{API_URL}/admin/llm_settings", timeout=10)
            if r.ok:
                data = r.json()
                self.provider.setText(data.get("provider") or "llama-cpp")
                self.model_path.setText(data.get("model_path") or "")
                self.device.setText("gpu")
                self.n_gpu_layers.setText(str(data.get("n_gpu_layers") or "-1"))
                self.n_threads.setText(str(data.get("n_threads") or "4"))
        except Exception:
            pass

    def save(self) -> None:
        payload = {
            "provider": self.provider.text().strip() or "llama-cpp",
            "model_path": self.model_path.text().strip() or None,
            "device": "gpu",
            "n_gpu_layers": int(self.n_gpu_layers.text()) if self.n_gpu_layers.text().strip() else -1,
            "n_threads": int(self.n_threads.text()) if self.n_threads.text().strip() else 4,
        }
        try:
            r = requests.post(f"{API_URL}/admin/llm_settings", json=payload, timeout=10)
            if r.ok:
                QMessageBox.information(self, "保存", "LLM設定を保存しました")
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))


class GlobalRulesPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.selected_id: Optional[int] = None

        layout = QHBoxLayout()

        # Left: list and NL instruction
        left = QVBoxLayout()
        instr_label = QLabel("自然言語の指示からルール追加")
        self.instr_edit = QLineEdit()
        btn_add_instr = QPushButton("指示を解析して追加")
        btn_add_instr.clicked.connect(self.add_from_instruction)  # type: ignore[arg-type]
        left.addWidget(instr_label)
        left.addWidget(self.instr_edit)
        left.addWidget(btn_add_instr)
        self.list = QListWidget()
        self.list.itemSelectionChanged.connect(self.on_select)  # type: ignore[arg-type]
        left.addWidget(QLabel("ルール一覧"))
        left.addWidget(self.list)
        btn_del = QPushButton("選択ルールを削除")
        btn_del.clicked.connect(self.delete_selected)  # type: ignore[arg-type]
        left.addWidget(btn_del)
        left.addStretch(1)

        # Right: form
        right = QVBoxLayout()
        form = QFormLayout()
        self.keyword = QLineEdit()
        self.debit = QLineEdit()
        self.credit = QLineEdit()
        self.priority = QLineEdit("0")
        self.enabled = QCheckBox("有効")
        self.enabled.setChecked(True)
        form.addRow("キーワード", self.keyword)
        form.addRow("借方科目", self.debit)
        form.addRow("貸方科目", self.credit)
        form.addRow("優先度", self.priority)
        form.addRow(" ", self.enabled)
        btns = QHBoxLayout()
        btn_new = QPushButton("新規")
        btn_save = QPushButton("保存/更新")
        btn_new.clicked.connect(self.clear_form)  # type: ignore[arg-type]
        btn_save.clicked.connect(self.save)  # type: ignore[arg-type]
        btns.addWidget(btn_new)
        btns.addWidget(btn_save)
        right.addLayout(form)
        right.addLayout(btns)
        right.addStretch(1)

        layout.addLayout(left, 2)
        layout.addLayout(right, 3)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        try:
            r = requests.get(f"{API_URL}/admin/global_rules", timeout=10)
            self.list.clear()
            if r.ok:
                for row in r.json():
                    item = QListWidgetItem(
                        f"prio={row['priority']} [{'ON' if row['enabled'] else 'OFF'}] {row['keyword']} => {row['debit_account']} / {row['credit_account']}"
                    )
                    item.setData(256, row)  # Qt.UserRole
                    self.list.addItem(item)
        except Exception:
            pass

    def clear_form(self) -> None:
        self.selected_id = None
        self.keyword.setText("")
        self.debit.setText("")
        self.credit.setText("")
        self.priority.setText("0")
        self.enabled.setChecked(True)

    def on_select(self) -> None:
        items = self.list.selectedItems()
        if not items:
            self.clear_form()
            return
        data = items[0].data(256) or {}
        self.selected_id = int(data.get("id"))
        self.keyword.setText(data.get("keyword") or "")
        self.debit.setText(data.get("debit_account") or "")
        self.credit.setText(data.get("credit_account") or "")
        self.priority.setText(str(data.get("priority") or 0))
        self.enabled.setChecked(bool(data.get("enabled")))

    def save(self) -> None:
        try:
            payload = {
                "id": self.selected_id,
                "keyword": self.keyword.text().strip(),
                "debit_account": self.debit.text().strip(),
                "credit_account": self.credit.text().strip(),
                "priority": int(self.priority.text()) if self.priority.text().strip() else 0,
                "enabled": bool(self.enabled.isChecked()),
            }
            if not payload["keyword"]:
                QMessageBox.warning(self, "検証エラー", "キーワードを入力してください")
                return
            r = requests.post(f"{API_URL}/admin/global_rules", json=payload, timeout=10)
            if r.ok:
                QMessageBox.information(self, "保存", "ルールを保存しました")
                self.refresh()
                self.clear_form()
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))

    def delete_selected(self) -> None:
        items = self.list.selectedItems()
        if not items:
            return
        data = items[0].data(256) or {}
        rid = data.get("id")
        if rid is None:
            return
        try:
            r = requests.delete(f"{API_URL}/admin/global_rules/{rid}", timeout=10)
            if r.ok:
                QMessageBox.information(self, "削除", "ルールを削除しました")
                self.refresh()
                self.clear_form()
            else:
                QMessageBox.warning(self, "削除失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "削除失敗", str(e))

    def add_from_instruction(self) -> None:
        text = self.instr_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "入力エラー", "指示を入力してください")
            return
        try:
            r = requests.post(f"{API_URL}/admin/nl_global_rule", json={"instruction": text}, timeout=10)
            if r.ok:
                QMessageBox.information(self, "追加", "指示からルールを追加しました")
                self.refresh()
                self.instr_edit.setText("")
            else:
                QMessageBox.warning(self, "追加失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "追加失敗", str(e))


class AdminWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("管理者ページ")
        self.stack = QStackedWidget()
        self.ocr_page = OCRStatusPage()
        self.llm_page = LLMSettingsPage()
        self.stack.addWidget(self.ocr_page)
        self.stack.addWidget(self.llm_page)
        self.rules_page = GlobalRulesPage()
        self.stack.addWidget(self.rules_page)
        self.setCentralWidget(self.stack)
        # Child window holder to prevent GC
        self._children = []
        menubar = self.menuBar()
        act_company = menubar.addAction("会社選択")
        act_company.triggered.connect(self.choose_company)  # type: ignore[arg-type]
        act_rules = menubar.addAction("自動仕訳ルール")
        act_ocr = menubar.addAction("PaddleOCR")
        act_llm = menubar.addAction("LLM設定")
        act_ocr.triggered.connect(lambda: self.stack.setCurrentIndex(0))  # type: ignore[arg-type]
        act_llm.triggered.connect(lambda: self.stack.setCurrentIndex(1))  # type: ignore[arg-type]
        act_rules.triggered.connect(lambda: self.stack.setCurrentIndex(2))  # type: ignore[arg-type]

    def choose_company(self) -> None:
        try:
            from .app import CompanySelector, MainWindow
        except Exception:
            return
        try:
            dlg = CompanySelector()
            res = dlg.exec()
            if res == QDialog.DialogCode.Accepted and getattr(dlg, 'selected', None):
                win = MainWindow(dlg.selected)
                win.resize(1200, 720)
                win.show()
                try:
                    self._children.append(win)  # retain reference
                except Exception:
                    pass
                try:
                    self.close()
                except Exception:
                    pass
        except Exception:
            pass

def create_admin_window() -> AdminWindow:
    win = AdminWindow()
    win.resize(900, 600)
    return win


class GlobalRulesWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("自動仕訳ルール 管理")
        self.page = GlobalRulesPage()
        self.setCentralWidget(self.page)


def create_rules_window() -> GlobalRulesWindow:
    win = GlobalRulesWindow()
    win.resize(900, 600)
    return win
