from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QDateEdit,
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
import requests
import time
from ..scansnap_control import reserve_and_scan
from .admin import create_admin_window


API_URL = "http://127.0.0.1:8765"


class CompanySelector(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("会社選択 / 新規作成")
        layout = QVBoxLayout()

        # 既存の会社から選択
        existing_group = QGroupBox("既存の会社から選択")
        eg_layout = QHBoxLayout()
        self.combo = QComboBox()
        refresh_btn = QPushButton("更新")
        select_btn = QPushButton("選択")
        refresh_btn.clicked.connect(self.load_companies)  # type: ignore[arg-type]
        select_btn.clicked.connect(self.on_select)  # type: ignore[arg-type]
        eg_layout.addWidget(self.combo)
        eg_layout.addWidget(refresh_btn)
        eg_layout.addWidget(select_btn)
        existing_group.setLayout(eg_layout)

        # 新規作成
        new_group = QGroupBox("新規作成")
        ng_layout = QHBoxLayout()
        self.company_input = QLineEdit()
        self.company_input.setPlaceholderText("新しい会社名を入力")
        create_btn = QPushButton("作成")
        create_btn.clicked.connect(self.on_create)  # type: ignore[arg-type]
        ng_layout.addWidget(self.company_input)
        ng_layout.addWidget(create_btn)
        new_group.setLayout(ng_layout)

        layout.addWidget(existing_group)
        layout.addWidget(new_group)

        # Admin button
        admin_btn = QPushButton("管理者")
        admin_btn.clicked.connect(self.on_admin)  # type: ignore[arg-type]
        layout.addWidget(admin_btn)
        self.setLayout(layout)

        self.selected: Optional[str] = None
        self.admin_requested: bool = False
        self.load_companies()

    def load_companies(self) -> None:
        self.combo.clear()
        try:
            resp = requests.get(f"{API_URL}/companies", timeout=5)
            names = [c.get("name") for c in resp.json() if isinstance(c, dict) and c.get("name")]
            for n in sorted(set(names)):
                self.combo.addItem(str(n))
        except Exception as e:
            QMessageBox.warning(self, "取得失敗", f"会社一覧の取得に失敗しました: {e}")

    def on_select(self) -> None:
        name = (self.combo.currentText() or "").strip()
        if not name:
            QMessageBox.warning(self, "選択エラー", "会社を選択してください")
            return
        self.selected = name
        self.accept()

    def on_create(self) -> None:
        name = self.company_input.text().strip()
        if not name:
            QMessageBox.warning(self, "入力エラー", "会社名を入力してください")
            return
        try:
            requests.post(f"{API_URL}/companies", json={"name": name}, timeout=5)
            self.selected = name
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"会社作成に失敗: {e}")

    def on_admin(self) -> None:
        # Signal admin mode to the bootstrap and close this dialog
        self.admin_requested = True
        self.accept()


class SettingsPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        # 勘定科目設定（補助科目・摘要候補）
        acct_group = QGroupBox("勘定科目設定（補助科目・摘要候補）")
        ag = QFormLayout()
        self.acct_name_edit = QLineEdit()
        self.acct_subs_edit = QLineEdit()
        self.acct_summaries_edit = QLineEdit()
        ag.addRow("勘定科目名", self.acct_name_edit)
        ag.addRow("補助科目（カンマ区切り）", self.acct_subs_edit)
        ag.addRow("摘要候補（カンマ区切り）", self.acct_summaries_edit)
        abtns = QHBoxLayout()
        acct_save = QPushButton("保存")
        acct_load = QPushButton("読込")
        acct_save.clicked.connect(self.on_save_account_setting)  # type: ignore[arg-type]
        acct_load.clicked.connect(self.on_load_account_setting)  # type: ignore[arg-type]
        abtns.addWidget(acct_save)
        abtns.addWidget(acct_load)
        ag.addRow(abtns)
        acct_group.setLayout(ag)
        layout.addWidget(acct_group)
        layout.addWidget(QLabel("設定"))
        # Output default config
        out_group = QGroupBox("出力先の既定設定")
        og = QVBoxLayout()
        self.default_hint = QLabel("出力ページに初期表示される既定の出力先を会社ごとに設定できます。参照は出力ページで行います。")
        og.addWidget(self.default_hint)
        form = QFormLayout()
        base = Path(__file__).resolve().parents[3]
        self.default_label = QLabel(f"app/output ({base / 'output'})")
        self.custom_default_edit = QLineEdit()
        form.addRow("既定: app/output", self.default_label)
        form.addRow("カスタム既定パス", self.custom_default_edit)
        # Archive base dir
        self.archive_label = QLabel("ラベル付け複製の保存ベースフォルダ（会社別）")
        self.archive_base_edit = QLineEdit()
        form.addRow(self.archive_label, self.archive_base_edit)
        og.addLayout(form)
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.on_save)  # type: ignore[arg-type]
        og.addWidget(save_btn)
        out_group.setLayout(og)
        layout.addWidget(out_group)

        # LLM override settings per company
        llm_group = QGroupBox("LLM設定（会社別オーバーライド）")
        lg = QVBoxLayout()
        self.llm_use_override = QLineEdit("0")
        self.llm_provider = QLineEdit("llama-cpp")
        self.llm_model_path = QLineEdit()
        self.llm_device = QLineEdit("cpu")
        self.llm_n_gpu_layers = QLineEdit("0")
        self.llm_n_threads = QLineEdit("4")
        self.llm_lora_path = QLineEdit()
        self.llm_prompt_template = QLineEdit()
        llm_form = QFormLayout()
        llm_form.addRow("Override使用(1/0)", self.llm_use_override)
        llm_form.addRow("Provider", self.llm_provider)
        llm_form.addRow("Model(GGUF)", self.llm_model_path)
        llm_form.addRow("Device(cpu/gpu)", self.llm_device)
        llm_form.addRow("GPU Layers", self.llm_n_gpu_layers)
        llm_form.addRow("Threads", self.llm_n_threads)
        llm_form.addRow("LoRA Path", self.llm_lora_path)
        llm_form.addRow("Prompt Template", self.llm_prompt_template)
        llm_save = QPushButton("LLM設定を保存")
        llm_save.clicked.connect(self.on_save_llm)  # type: ignore[arg-type]
        lg.addLayout(llm_form)
        lg.addWidget(llm_save)
        llm_group.setLayout(lg)
        layout.addWidget(llm_group)

        # LLM logs quick view
        logs_group = QGroupBox("LLMログ（最近）")
        lgl = QVBoxLayout()
        self.llm_logs = QListWidget()
        refresh = QPushButton("ログ更新")
        refresh.clicked.connect(self.load_llm_logs)  # type: ignore[arg-type]
        lgl.addWidget(self.llm_logs)
        lgl.addWidget(refresh)
        logs_group.setLayout(lgl)
        layout.addWidget(logs_group)
        layout.addStretch(1)
        self.setLayout(layout)
        self.load_settings()
        self.load_llm_settings()
        self.load_llm_logs()

    def on_save(self) -> None:
        path = self.custom_default_edit.text().strip() or None
        archive_dir = self.archive_base_edit.text().strip() or None
        try:
            r = requests.post(
                f"{API_URL}/settings",
                json={"company_name": self.company, "default_output_dir": path, "archive_base_dir": archive_dir},
                timeout=10,
            )
            if r.ok:
                QMessageBox.information(self, "保存", "設定を保存しました")
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))

    def load_settings(self) -> None:
        try:
            r = requests.get(f"{API_URL}/settings", params={"company_name": self.company}, timeout=10)
            if r.ok:
                data = r.json()
                self.custom_default_edit.setText(data.get("default_output_dir") or "")
                self.archive_base_edit.setText(data.get("archive_base_dir") or "")
        except Exception:
            pass

    def on_save_account_setting(self) -> None:
        name = self.acct_name_edit.text().strip()
        subs = [s.strip() for s in self.acct_subs_edit.text().split(',') if s.strip()]
        sums = [s.strip() for s in self.acct_summaries_edit.text().split(',') if s.strip()]
        if not name:
            QMessageBox.warning(self, "入力エラー", "勘定科目名を入力してください")
            return
        try:
            r = requests.post(
                f"{API_URL}/account_settings",
                json={"company_name": self.company, "account_name": name, "subaccounts": subs, "summaries": sums},
                timeout=10,
            )
            if r.ok:
                QMessageBox.information(self, "保存", "勘定科目設定を保存しました")
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))

    def on_load_account_setting(self) -> None:
        name = self.acct_name_edit.text().strip()
        try:
            r = requests.get(f"{API_URL}/account_settings", params={"company_name": self.company}, timeout=10)
            if not r.ok:
                return
            rows = r.json() or []
            if not name:
                if rows:
                    row = rows[0]
                    self.acct_name_edit.setText(row.get("account_name") or "")
                    self.acct_subs_edit.setText(
                        ", ".join(row.get("subaccounts") or [])
                    )
                    self.acct_summaries_edit.setText(
                        ", ".join(row.get("summaries") or [])
                    )
                return
            for row in rows:
                if row.get("account_name") == name:
                    self.acct_subs_edit.setText(
                        ", ".join(row.get("subaccounts") or [])
                    )
                    self.acct_summaries_edit.setText(
                        ", ".join(row.get("summaries") or [])
                    )
                    break
        except Exception:
            pass

    def load_llm_settings(self) -> None:
        try:
            r = requests.get(f"{API_URL}/company_llm_settings", params={"company_name": self.company}, timeout=10)
            if r.ok:
                data = r.json()
                self.llm_use_override.setText("1" if data.get("use_override") else "0")
                self.llm_provider.setText(data.get("provider") or "llama-cpp")
                self.llm_model_path.setText(data.get("model_path") or "")
                self.llm_device.setText(data.get("device") or "cpu")
                self.llm_n_gpu_layers.setText(str(data.get("n_gpu_layers") or "0"))
                self.llm_n_threads.setText(str(data.get("n_threads") or "4"))
                self.llm_lora_path.setText(data.get("lora_path") or "")
                self.llm_prompt_template.setText(data.get("prompt_template") or "")
        except Exception:
            pass

    def on_save_llm(self) -> None:
        try:
            payload = {
                "company_name": self.company,
                "use_override": self.llm_use_override.text().strip() == "1",
                "provider": self.llm_provider.text().strip() or "llama-cpp",
                "model_path": self.llm_model_path.text().strip() or None,
                "device": self.llm_device.text().strip() or "cpu",
                "n_gpu_layers": int(self.llm_n_gpu_layers.text()) if self.llm_n_gpu_layers.text().strip() else 0,
                "n_threads": int(self.llm_n_threads.text()) if self.llm_n_threads.text().strip() else 4,
                "lora_path": self.llm_lora_path.text().strip() or None,
                "prompt_template": self.llm_prompt_template.text().strip() or None,
            }
            r = requests.post(f"{API_URL}/company_llm_settings", json=payload, timeout=10)
            if r.ok:
                QMessageBox.information(self, "保存", "LLM設定を保存しました")
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))

    def load_llm_logs(self) -> None:
        try:
            r = requests.get(f"{API_URL}/company_llm_logs", params={"company_name": self.company, "limit": 50}, timeout=10)
            if r.ok:
                self.llm_logs.clear()
                for row in r.json():
                    self.llm_logs.addItem(
                        f"doc#{row['document_id']} conf={row.get('confidence')} {row.get('created_at')} model={row.get('model_id')}"
                    )
        except Exception:
            pass


class OutputPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company

        layout = QVBoxLayout()
        # Destination chooser (prefilled from settings)
        dest_row = QHBoxLayout()
        self.dest_edit = QLineEdit()
        browse = QPushButton("参照…")
        browse.clicked.connect(self.choose_dir)  # type: ignore[arg-type]
        dest_row.addWidget(QLabel("出力先フォルダ"))
        dest_row.addWidget(self.dest_edit)
        dest_row.addWidget(browse)

        self.info = QLabel()
        export_btn = QPushButton("弥生インポート形式で出力")
        export_btn.clicked.connect(self.on_export)  # type: ignore[arg-type]

        # Past exports list
        self.list_widget = QListWidget()
        refresh_btn = QPushButton("履歴を更新")
        refresh_btn.clicked.connect(self.refresh_history)  # type: ignore[arg-type]

        layout.addLayout(dest_row)
        layout.addWidget(self.info)
        layout.addWidget(export_btn)
        layout.addWidget(QLabel("過去の出力"))
        layout.addWidget(self.list_widget)
        layout.addWidget(refresh_btn)
        layout.addStretch(1)
        self.setLayout(layout)
        self.update_info()
        self.refresh_history()

    def update_info(self) -> None:
        # Pre-fill destination based on company settings
        base = Path(__file__).resolve().parents[3]
        default_dir = base / "output"
        try:
            r = requests.get(f"{API_URL}/settings", params={"company_name": self.company}, timeout=10)
            if r.ok:
                data = r.json()
                cd = data.get("default_output_dir") or str(default_dir)
                self.dest_edit.setText(cd)
                if data.get("default_output_dir"):
                    self.info.setText(f"既定: カスタム ({cd})")
                else:
                    self.info.setText(f"既定: デフォルト ({default_dir})")
            else:
                self.dest_edit.setText(str(default_dir))
                self.info.setText(f"既定: デフォルト ({default_dir})")
        except Exception:
            self.dest_edit.setText(str(default_dir))
            self.info.setText(f"既定: デフォルト ({default_dir})")

    def on_export(self) -> None:
        dest = self.dest_edit.text().strip()
        params = {"company_name": self.company}
        if dest:
            params["target_dir"] = dest
        try:
            r = requests.post(f"{API_URL}/export", params=params, timeout=60)
            if r.ok:
                path = r.json().get("csv")
                QMessageBox.information(self, "出力完了", f"出力しました:\n{path}")
                self.refresh_history()
            else:
                QMessageBox.warning(self, "出力失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "出力失敗", str(e))

    def choose_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "出力先フォルダ選択", self.dest_edit.text() or str(Path.home()))
        if d:
            self.dest_edit.setText(d)

    def _history_dir(self) -> Path:
        p = Path(self.dest_edit.text().strip())
        if p.exists() and p.is_dir():
            return p
        base = Path(__file__).resolve().parents[3]
        return base / "output"

    def refresh_history(self) -> None:
        self.list_widget.clear()
        d = self._history_dir()
        try:
            for f in sorted(d.glob("*_yayoi.csv"), key=lambda x: x.stat().st_mtime, reverse=True)[:200]:
                self.list_widget.addItem(str(f))
        except Exception:
            pass


class ScanPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        # ScanSnap watch folder and scan button
        folder_row = QHBoxLayout()
        self.watch_folder = QLineEdit()
        default_folder = Path.home() / "Documents" / "ScanSnap"
        self.watch_folder.setText(str(default_folder))
        browse_btn = QPushButton("参照…")
        browse_btn.clicked.connect(self.choose_folder)  # type: ignore[arg-type]
        folder_row.addWidget(QLabel("保存先フォルダ"))
        folder_row.addWidget(self.watch_folder)
        folder_row.addWidget(browse_btn)
        self.scan_btn = QPushButton("スキャン開始")
        self.scan_btn.clicked.connect(self.start_scan)  # type: ignore[arg-type]
        layout.addLayout(folder_row)
        layout.addWidget(self.scan_btn)

        import_btn = QPushButton("PDF取込")
        import_btn.clicked.connect(self.import_pdfs)  # type: ignore[arg-type]
        self.list_widget = QListWidget()

        layout.addWidget(import_btn)
        layout.addWidget(QLabel("未確認データ"))
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

        # State for polling scanned files
        self._poll_timer: QTimer | None = None
        self._known_files: set[str] = set()
        self._scan_started_at: float = 0.0
        self._poll_deadline: float = 0.0

        self.refresh()

    def refresh(self) -> None:
        self.list_widget.clear()
        try:
            resp = requests.get(f"{API_URL}/documents", params={"company_name": self.company, "status": "unconfirmed"}, timeout=10)
            for item in resp.json():
                lw = QListWidgetItem(f"#{item['id']} - {Path(item['file_path']).name}")
                lw.setData(Qt.ItemDataRole.UserRole, item)
                self.list_widget.addItem(lw)
        except Exception:
            pass

    def import_pdfs(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "PDFを選択", str(Path.cwd()), "PDF Files (*.pdf)")
        for f in files:
            try:
                with open(f, "rb") as fh:
                    files_ = {"file": (Path(f).name, fh, "application/pdf")}
                    data = {"company_name": self.company}
                    requests.post(f"{API_URL}/documents/import", data=data, files=files_, timeout=60)
            except Exception as e:
                QMessageBox.warning(self, "取込失敗", f"{f}: {e}")
        self.refresh()

    # Added: select watch folder for scanned PDFs
    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "フォルダ選択", self.watch_folder.text() or str(Path.home()))
        if folder:
            self.watch_folder.setText(folder)

    # Added: start ScanSnap scan via SDK and poll folder for new PDFs
    def start_scan(self) -> None:
        self.scan_btn.setEnabled(False)
        folder = Path(self.watch_folder.text())
        folder.mkdir(parents=True, exist_ok=True)
        self._known_files = {p.name for p in folder.glob("*.pdf")}
        self._scan_started_at = time.time()

        ok = False
        try:
            ok = reserve_and_scan()
        except Exception as e:
            QMessageBox.warning(self, "スキャン開始失敗", str(e))

        if not ok:
            QMessageBox.warning(self, "スキャン開始失敗", "ScanSnap Homeを起動できませんでした。")
            self.scan_btn.setEnabled(True)
            return

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_folder)  # type: ignore[arg-type]
        self._poll_deadline = time.time() + 120
        self._poll_timer.start(1000)

    def _poll_folder(self) -> None:
        folder = Path(self.watch_folder.text())
        found_new = False
        for p in folder.glob("*.pdf"):
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            if p.name in getattr(self, "_known_files", set()):
                continue
            if st.st_mtime >= getattr(self, "_scan_started_at", 0.0):
                try:
                    with open(p, "rb") as fh:
                        files_ = {"file": (p.name, fh, "application/pdf")}
                        data = {"company_name": self.company}
                        requests.post(f"{API_URL}/documents/import", data=data, files=files_, timeout=60)
                    self._known_files.add(p.name)
                    found_new = True
                except Exception:
                    pass
        if found_new:
            self.refresh()
        if time.time() > getattr(self, "_poll_deadline", 0.0):
            if getattr(self, "_poll_timer", None):
                self._poll_timer.stop()
                self._poll_timer = None
            self.scan_btn.setEnabled(True)


class ReviewPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()

        splitter = QSplitter()
        self.left = QListWidget()
        self.left.itemSelectionChanged.connect(self.on_selected)  # type: ignore[arg-type]

        right = QWidget()
        form = QFormLayout()
        self.date = QDateEdit()
        self.date.setDisplayFormat("yyyy/MM/dd")
        self.date.setCalendarPopup(True)
        self.amount = QLineEdit()
        self.summary = QComboBox()
        self.summary.setEditable(True)
        self.debit = QLineEdit()
        self.credit = QLineEdit()
        form.addRow("日付(YYYY/MM/DD)", self.date)
        form.addRow("金額", self.amount)
        form.addRow("摘要", self.summary)
        form.addRow("借方勘定", self.debit)
        form.addRow("貸方勘定", self.credit)
        # extra fields: subaccounts + invoice status
        self.debit_sub = QComboBox(); self.debit_sub.setEditable(True)
        self.credit_sub = QComboBox(); self.credit_sub.setEditable(True)
        self.invoice = QComboBox(); self.invoice.addItems(["", "適格", "非適格", "非課税"])
        form.addRow("借方補助科目", self.debit_sub)
        form.addRow("貸方補助科目", self.credit_sub)
        form.addRow("請求書区分(インボイス)", self.invoice)
        right.setLayout(form)

        # Load account setting options for subaccount/summary
        self._acct_settings = {}
        try:
            r = requests.get(f"{API_URL}/account_settings", params={"company_name": self.company}, timeout=10)
            if r.ok:
                for row in r.json():
                    self._acct_settings[row.get("account_name")] = {
                        "subaccounts": row.get("subaccounts") or [],
                        "summaries": row.get("summaries") or [],
                    }
        except Exception:
            pass
        self.debit.textChanged.connect(lambda _=None: self._populate_from_accounts())  # type: ignore[arg-type]
        self.credit.textChanged.connect(lambda _=None: self._populate_from_accounts())  # type: ignore[arg-type]

        splitter.addWidget(self.left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        btn_row = QHBoxLayout()
        later_btn = QPushButton("後で確認")
        ok_btn = QPushButton("OK")
        del_btn = QPushButton("削除")
        later_btn.clicked.connect(self.on_check_later)  # type: ignore[arg-type]
        ok_btn.clicked.connect(self.on_ok)  # type: ignore[arg-type]
        del_btn.clicked.connect(self.on_delete)  # type: ignore[arg-type]
        btn_row.addWidget(later_btn)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(del_btn)

        layout.addWidget(splitter)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        self.refresh()

    def _populate_from_accounts(self) -> None:
        try:
            deb = self.debit.text().strip()
            cre = self.credit.text().strip()
            deb_opts = (self._acct_settings.get(deb) or {}).get("subaccounts") or []
            cre_opts = (self._acct_settings.get(cre) or {}).get("subaccounts") or []
            # Update subaccount comboboxes
            self.debit_sub.clear(); self.debit_sub.addItems([""] + list(deb_opts))
            self.credit_sub.clear(); self.credit_sub.addItems([""] + list(cre_opts))
            # Update summary options (use debit account as anchor)
            sums = (self._acct_settings.get(deb) or {}).get("summaries") or []
            self.summary.clear(); self.summary.addItems([""] + list(sums))
        except Exception:
            pass

    def refresh(self) -> None:
        self.left.clear()
        try:
            resp = requests.get(f"{API_URL}/documents", params={"company_name": self.company, "status": "unconfirmed"}, timeout=10)
            for item in resp.json():
                lw = QListWidgetItem(f"#{item['id']} - {Path(item['file_path']).name}")
                lw.setData(Qt.ItemDataRole.UserRole, item)
                self.left.addItem(lw)
        except Exception:
            pass

    def on_selected(self) -> None:
        items = self.left.selectedItems()
        if not items:
            return
        item = items[0].data(Qt.ItemDataRole.UserRole)
        auto = item.get("auto", {})
        try:
            d = auto.get("date") or ""
            if d and "/" in d:
                y, m, da = [int(x) for x in d.split("/")]
                self.date.setDate(QDate(y, m, da))
            else:
                self.date.setDate(QDate.currentDate())
        except Exception:
            self.date.setDate(QDate.currentDate())
        self.amount.setText(str(auto.get("amount") or ""))
        self.summary.setEditText(auto.get("summary") or "")
        self.debit.setText(auto.get("debit_account") or "")
        self.credit.setText(auto.get("credit_account") or "")
        # populate and set subaccounts and invoice
        self._populate_from_accounts()
        self.debit_sub.setCurrentText(auto.get("debit_subaccount") or "")
        self.credit_sub.setCurrentText(auto.get("credit_subaccount") or "")
        self.invoice.setCurrentText(auto.get("invoice_status") or "")

    def _current_doc_id(self) -> Optional[int]:
        items = self.left.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.ItemDataRole.UserRole)["id"]

    def on_check_later(self) -> None:
        doc_id = self._current_doc_id()
        if not doc_id:
            return
        requests.post(f"{API_URL}/documents/{doc_id}/check", timeout=10)
        self.refresh()

    def on_ok(self) -> None:
        doc_id = self._current_doc_id()
        if not doc_id:
            return
        payload = {
            "date": self.date.text().strip() or None,
            "amount": int(self.amount.text()) if self.amount.text().strip() else None,
            "summary": self.summary.currentText().strip() or None,
            "debit_account": self.debit.text().strip() or None,
            "credit_account": self.credit.text().strip() or None,
            "debit_subaccount": self.debit_sub.currentText().strip() or None,
            "credit_subaccount": self.credit_sub.currentText().strip() or None,
            "invoice_status": self.invoice.currentText().strip() or None,
        }
        r = requests.post(f"{API_URL}/documents/{doc_id}/ok", json=payload, timeout=20)
        if r.ok:
            QMessageBox.information(self, "保存", "確定・学習しました")
        else:
            QMessageBox.warning(self, "エラー", r.text)
        self.refresh()

    def on_delete(self) -> None:
        doc_id = self._current_doc_id()
        if not doc_id:
            return
        requests.delete(f"{API_URL}/documents/{doc_id}", timeout=10)
        self.refresh()


class CheckPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        self.list_widget.clear()
        try:
            resp = requests.get(f"{API_URL}/documents", params={"company_name": self.company, "status": "check_later"}, timeout=10)
            for item in resp.json():
                self.list_widget.addItem(f"#{item['id']} - {Path(item['file_path']).name}")
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
        self.check_page = CheckPage(company)
        self.output_page = OutputPage(company)
        self.settings_page = SettingsPage(company)
        self.stack.addWidget(self.scan_page)
        self.stack.addWidget(self.review_page)
        self.stack.addWidget(self.check_page)
        self.stack.addWidget(self.output_page)
        self.stack.addWidget(self.settings_page)
        self.setCentralWidget(self.stack)

        menubar = self.menuBar()
        # Add top-level actions on menu bar
        # Company selector first
        act_company = menubar.addAction("会社選択")
        act_scan = menubar.addAction("スキャン")
        act_review = menubar.addAction("仕訳確認")
        act_check = menubar.addAction("要チェック")
        act_export = menubar.addAction("出力")
        act_settings = menubar.addAction("設定")

        act_company.triggered.connect(self.change_company)  # type: ignore[arg-type]
        act_scan.triggered.connect(self.show_scan)  # type: ignore[arg-type]
        act_review.triggered.connect(self.show_review)  # type: ignore[arg-type]
        act_check.triggered.connect(self.show_check)  # type: ignore[arg-type]
        act_export.triggered.connect(self.show_output)  # type: ignore[arg-type]
        act_settings.triggered.connect(self.show_settings)  # type: ignore[arg-type]

    def show_scan(self) -> None:
        self.stack.setCurrentIndex(0)

    def show_review(self) -> None:
        self.stack.setCurrentIndex(1)

    def show_check(self) -> None:
        self.stack.setCurrentIndex(2)

    def show_output(self) -> None:
        # Update label to reflect any settings change
        self.output_page.update_info()
        self.stack.setCurrentIndex(3)

    def show_settings(self) -> None:
        self.stack.setCurrentIndex(4)

    def change_company(self) -> None:
        dlg = CompanySelector()
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected:
            new_company = dlg.selected
            if new_company and new_company != self.company:
                self.company = new_company
                # Rebuild pages bound to company
                new_stack = QStackedWidget()
                self.scan_page = ScanPage(self.company)
                self.review_page = ReviewPage(self.company)
                self.check_page = CheckPage(self.company)
                self.output_page = OutputPage(self.company)
                self.settings_page = SettingsPage(self.company)
                new_stack.addWidget(self.scan_page)
                new_stack.addWidget(self.review_page)
                new_stack.addWidget(self.check_page)
                new_stack.addWidget(self.output_page)
                new_stack.addWidget(self.settings_page)
                self.setCentralWidget(new_stack)
                self.stack = new_stack


def run_ui() -> None:
    app = QApplication(sys.argv)

    # Ask company
    selector = CompanySelector()
    result = selector.exec()
    # If admin requested from selector, launch admin window
    if selector.admin_requested:
        try:
            admin_win = create_admin_window()
            admin_win.show()
            sys.exit(app.exec())
        except Exception:
            return

    company = selector.selected if result == QDialog.DialogCode.Accepted else None
    if not company:
        return

    # Build window
    win = MainWindow(company)
    win.resize(1100, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_ui()
