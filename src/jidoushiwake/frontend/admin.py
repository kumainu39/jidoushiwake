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
)
import requests


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
        self.model_path = QLineEdit()
        self.device = QLineEdit("cpu")  # "cpu" or "gpu"
        self.n_gpu_layers = QLineEdit("0")
        self.n_threads = QLineEdit("4")
        form.addRow("Provider", self.provider)
        form.addRow("Model Path (GGUF)", self.model_path)
        form.addRow("Device (cpu/gpu)", self.device)
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
                self.device.setText(data.get("device") or "cpu")
                self.n_gpu_layers.setText(str(data.get("n_gpu_layers") or "0"))
                self.n_threads.setText(str(data.get("n_threads") or "4"))
        except Exception:
            pass

    def save(self) -> None:
        payload = {
            "provider": self.provider.text().strip() or "llama-cpp",
            "model_path": self.model_path.text().strip() or None,
            "device": self.device.text().strip() or "cpu",
            "n_gpu_layers": int(self.n_gpu_layers.text()) if self.n_gpu_layers.text().strip() else 0,
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


class AdminWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("管理者ページ")
        self.stack = QStackedWidget()
        self.ocr_page = OCRStatusPage()
        self.llm_page = LLMSettingsPage()
        self.stack.addWidget(self.ocr_page)
        self.stack.addWidget(self.llm_page)
        self.setCentralWidget(self.stack)
        menubar = self.menuBar()
        act_ocr = menubar.addAction("PaddleOCR")
        act_llm = menubar.addAction("LLM設定")
        act_ocr.triggered.connect(lambda: self.stack.setCurrentIndex(0))  # type: ignore[arg-type]
        act_llm.triggered.connect(lambda: self.stack.setCurrentIndex(1))  # type: ignore[arg-type]


def create_admin_window() -> AdminWindow:
    win = AdminWindow()
    win.resize(900, 600)
    return win
