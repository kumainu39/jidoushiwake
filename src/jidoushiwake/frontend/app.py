from __future__ import annotations

import sys
from functools import partial
import json
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QBrush, QColor, QPixmap, QImage
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
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDialog,
    QTextEdit,
    QSizePolicy,
    QScrollArea,
)
from PyQt6.QtWidgets import QCompleter, QStyledItemDelegate, QStyleOptionViewItem, QStyle
from PyQt6.QtCore import QSortFilterProxyModel
from PyQt6.QtGui import QStandardItemModel, QStandardItem
import requests
import time
from ..scansnap_control import reserve_and_scan
from .admin import create_admin_window


API_URL = "http://127.0.0.1:8765"


# UI configurable settings
def _load_ui_settings() -> dict:
    defaults = {
        "pdf_label_min_width": 480,
        "pdf_label_max_width": 0,  # 0 = no limit
        "pdf_fixed_width": 420,
        "pdf_scale_ratio": 1.25,
        "pdf_scale_ratio_min": 0.0,
        "pdf_scale_ratio_max": 2.0,
        "pdf_max_width_px": 0,  # 0 = auto
        "nl_group_min_height": 60,
        "nl_group_max_height": 0,
        "nl_edit_min_height": 120,
        "nl_edit_max_height": 0,
        "nl_reserved_height_px": 180,
        "left_container_min_width": 520,
        "left_container_max_width": 0,
        "left_scroll_min_width": 500,
        "left_scroll_max_width": 0,
        "splitter_stretch_left": 3,
        "splitter_stretch_right": 4,
        "splitter_min_left_px": 600,
        "splitter_left_ratio": 0.4,
        "splitter_max_left_px": 0,
        "splitter_left_max_ratio": 1.0,
        # New: right side target ratio and left side min ratio
        "splitter_right_ratio": 0.0,        # 0 = unused; if >0, left_ratio := 1-right_ratio
        "splitter_left_min_ratio": 0.0,     # lower bound ratio for left
        # New: allow exact left width specification
        "splitter_left_fixed_px": 0,
        "splitter_left_fixed_ratio": 0.0,
        # New: only apply splitter policy on maximize
        "apply_splitter_only_when_maximized": True,
    }
    path_env = os.environ.get("JIDOU_UI_SETTINGS")
    candidate_paths: list[Path] = []
    try:
        if path_env:
            candidate_paths.append(Path(path_env))
    except Exception:
        pass
    try:
        candidate_paths.append(Path.cwd() / "ui_settings.json")
        candidate_paths.append(Path.cwd() / "config" / "ui_settings.json")
        repo_root_guess = Path(__file__).resolve().parents[3]
        candidate_paths.append(repo_root_guess / "config" / "ui_settings.json")
    except Exception:
        pass
    for p in candidate_paths:
        try:
            if p.is_file():
                with p.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    merged = {**defaults, **data}
                    return merged
        except Exception:
            # On parse error or other issues, fall back to defaults silently
            break
    return defaults


UI_SETTINGS = _load_ui_settings()


def _clean_display_name(name: str) -> str | None:
    """Return a cleaned Japanese account name or None to drop it.

    Heuristics:
    - Remove ASCII/number suffixes (romaji/codes appended to the right).
    - Drop bracketed notes in ()/（） entirely.
    - Exclude headings like "資産の部/負債の部/純資産の部" and lines that contain "の部".
    - Exclude names that still contain ASCII letters/digits after cleanup.
    - Require at least one Japanese character (Kana/Kanji) and length >= 2.
    """
    try:
        import re
        s = (name or "").strip()
        if not s:
            return None
        # remove bracketed content
        s = re.sub(r"[（(][^）)]*[）)]", "", s)
        # cut off at first ASCII letter/number occurrence (and right side)
        m = re.match(r"^([^A-Za-z0-9]+)", s)
        if m:
            s = m.group(1).strip()
        # normalize spaces and punctuation
        s = re.sub(r"\s+", "", s)
        # headings like 資産の部/負債の部/純資産の部 などは除外
        if "の部" in s:
            return None
        if any(k in s for k in ("合計", "小計")):
            return None
        # require Japanese chars
        def has_jp(t: str) -> bool:
            return any((0x3040 <= ord(ch) <= 0x30FF) or (0x3400 <= ord(ch) <= 0x9FFF) for ch in t)
        if not has_jp(s):
            return None
        # exclude if ASCII/digits remain
        if any(("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch.isdigit() for ch in s):
            return None
        if len(s) < 2:
            return None
        return s
    except Exception:
        return name or None


def _load_account_catalog() -> tuple[list[str], dict[str, str]]:
    """Return (display_names, token_to_name).

    - display_names: 科目名のみ（日本語）。コードやローマ字は除外。
    - token_to_name: 検索用トークン（コード/ローマ字等）→ 表示名 へのマップ。
    """
    names: list[str] = []
    token_to_name: dict[str, str] = {}
    # API source
    try:
        r = requests.get(f"{API_URL}/admin/account_master", timeout=10)
        if r.ok:
            for row in r.json() or []:
                raw_name = (row or {}).get("name")
                code = (row or {}).get("code")
                if raw_name:
                    cn = _clean_display_name(str(raw_name))
                    if cn:
                        names.append(cn)
                    if code:
                        token_to_name[str(code)] = cn
    except Exception:
        pass
    # Seed fallback
    if not names:
        try:
            candidates = [
                Path.cwd() / "data" / "account_master_seed.txt",
                Path(__file__).resolve().parents[3] / "data" / "account_master_seed.txt",
            ]
        except Exception:
            candidates = []
        for sp in candidates:
            try:
                if not sp.is_file():
                    continue
                with sp.open("r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        raw = (line or "").strip()
                        if not raw or raw.startswith("#"):
                            continue
                        # CSV優先: code,name,romaji1,...
                        parts = [p.strip() for p in raw.split(",")]
                        if len(parts) > 1:
                            code, nm, *rest = parts
                            clean = _clean_display_name(nm)
                            if clean:
                                names.append(clean)
                                if code:
                                    token_to_name[code] = clean
                                for tok in rest:
                                    tok = (tok or '').strip()
                                    if tok:
                                        token_to_name[tok] = clean
                            continue
                        # 非CSV（空白区切り）: 日本語名 [ROMAJI] [CODE]
                        s = raw
                        # exclude section/group headings early
                        if "の部" in s:
                            continue
                        import re
                        m = re.match(r"^([^\x00-\x7F]+)\s*(.*)$", s)
                        if not m:
                            # no clear Japanese prefix; fallback to cleaning whole
                            nm = _clean_display_name(s)
                            if nm:
                                names.append(nm)
                            continue
                        jp, rest = m.group(1).strip(), m.group(2).strip()
                        clean = _clean_display_name(jp)
                        if not clean:
                            continue
                        toks: list[str] = []
                        if rest:
                            for w in re.split(r"\s+", rest):
                                w = (w or '').strip()
                                if not w:
                                    continue
                                # take ASCII tokens only
                                if all(ord(c) < 128 for c in w):
                                    toks.append(w)
                        names.append(clean)
                        for tok in toks:
                            token_to_name[tok] = clean
                if names:
                    break
            except Exception:
                continue
    # Final pass: keep only displayable Japanese names and clean again
    def _is_displayable(s: str) -> bool:
        return any((0x3040 <= ord(ch) <= 0x30FF) or (0x3400 <= ord(ch) <= 0x9FFF) for ch in s)
    names = [(_clean_display_name(n) or "") for n in names]
    names = [n for n in names if n and _is_displayable(n)]
    # de-duplicate preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    for n in uniq:
        token_to_name.setdefault(n, n)
        token_to_name.setdefault(n.lower(), n)
    # also index lowercase variants of provided tokens
    token_to_name = { (k or "").strip(): v for k, v in token_to_name.items() if (k or "").strip() }
    extra: dict[str, str] = {}
    for k, v in token_to_name.items():
        lk = k.lower()
        if lk not in token_to_name:
            extra[lk] = v
    token_to_name.update(extra)
    return uniq, token_to_name


def _load_account_names() -> list[str]:
    names, _ = _load_account_catalog()
    return names


class _AccountComboBox(QComboBox):
    def __init__(self, items: list[str], token_to_name: dict[str, str] | None = None) -> None:
        super().__init__()
        self._token_to_name = { (k or '').strip(): v for k, v in (token_to_name or {}).items() }
        self._token_to_name_lower = { (k or '').strip().lower(): v for k, v in self._token_to_name.items() }
        self._allow_index_change: bool = False
        try:
            self.setEditable(True)
            # Avoid default auto-completion that replaces the text after a single keystroke
            try:
                self.setCompleter(None)
            except Exception:
                pass
            try:
                self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            except Exception:
                pass
        except Exception:
            pass
        try:
            # Use a custom model that displays names only in the combo,
            # and a separate completer model that maps many tokens → the same name.
            model = QStandardItemModel()  # combo's display model (name only)
            names = list(items)
            # Gather tokens per name
            name_tokens: dict[str, set[str]] = {n: set() for n in names}
            for tok, nm in self._token_to_name.items():
                if not nm:
                    continue
                if nm not in name_tokens:
                    continue
                t = (tok or '').strip()
                if not t:
                    continue
                # sanitize tokens: ASCII only, len>=2 for alpha, >=2 for digits（2文字から検索したい要望に対応）
                if not all(ord(c) < 128 for c in t):
                    continue
                if t.isalpha() and len(t) < 2:
                    continue
                if t.isdigit() and len(t) < 2:
                    continue
                if len(t) < 2:
                    continue
                name_tokens[nm].add(t.lower())
            # Build items with DisplayRole=name, UserRole=search_blob, UserRole+1=right token, UserRole+2=tokens(list)
            from PyQt6.QtCore import Qt as _Qt
            for nm in names:
                it = QStandardItem(nm)
                toks = sorted(name_tokens.get(nm, set()))
                blob = ' '.join(toks + [nm.lower()])
                it.setData(blob, _Qt.ItemDataRole.UserRole)
                try:
                    it.setData(toks, _Qt.ItemDataRole.UserRole + 2)
                except Exception:
                    pass
                # choose representative token to render on the right (prefer alpha)
                rep = next((x.upper() for x in toks if x.isalpha()), '') or next((x for x in toks if x.isdigit()), '')
                it.setData(rep, _Qt.ItemDataRole.UserRole + 1)
                model.appendRow(it)
            # Install proxy filter so typing in the editor filters popup by tokens as well
            class _Proxy(QSortFilterProxyModel):
                def __init__(self) -> None:
                    super().__init__()
                    self._query = ''
                    self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                def setQuery(self, q: str) -> None:
                    self._query = (q or '').strip()
                    self.invalidateFilter()
                def filterAcceptsRow(self, src_row: int, src_parent) -> bool:  # type: ignore[override]
                    if not self._query:
                        return True
                    m = self.sourceModel()
                    try:
                        idx = m.index(src_row, 0, src_parent)
                        name = (m.data(idx) or '').lower()
                        blob = (m.data(idx, _Qt.ItemDataRole.UserRole) or '').lower()
                        toks = m.data(idx, _Qt.ItemDataRole.UserRole + 2) or []
                        q = self._query.lower()
                        if q in name or q in blob:
                            return True
                        for t in toks:
                            try:
                                if q in (t or '').lower():
                                    return True
                            except Exception:
                                continue
                        return False
                    except Exception:
                        return True
            proxy = _Proxy()
            proxy.setSourceModel(model)
            self.setModel(proxy)
            # Prevent popup view from stealing focus while typing
            try:
                from PyQt6.QtCore import Qt as _Qt
                self.view().setFocusPolicy(_Qt.FocusPolicy.NoFocus)
            except Exception:
                pass
            # Filter as user types and keep popup open
            try:
                le = self.lineEdit()
                if le is not None:
                    # Ensure normal echo so typed characters are visible
                    try:
                        from PyQt6.QtWidgets import QLineEdit as _QLineEdit
                        le.setEchoMode(_QLineEdit.EchoMode.Normal)
                    except Exception:
                        pass
                    def _on_edit(txt: str) -> None:
                        try:
                            import unicodedata as _ud
                            norm = _ud.normalize('NFKC', txt)
                            proxy.setQuery(norm)
                            if not self.view().isVisible():
                                self.showPopup()
                            try:
                                from PyQt6.QtCore import Qt as _Qt
                                le.setFocus(_Qt.FocusReason.OtherFocusReason)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    le.textEdited.connect(_on_edit)  # type: ignore[arg-type]
            except Exception:
                pass
        except Exception:
            pass
        # Normalize when user finishes typing without selecting a completion
        try:
            le = self.lineEdit()
            if le is not None:
                try:
                    le.editingFinished.connect(self._apply_normalization)  # type: ignore[arg-type]
                except Exception:
                    pass
        except Exception:
            pass

    def _apply_normalization(self) -> None:
        try:
            txt = (self.currentText() or "").strip()
            mapped = self._best_map(txt)
            if mapped != self.currentText():
                self.setCurrentText(mapped)
        except Exception:
            pass

    # Guard against implicit index changes while editing
    def setCurrentIndex(self, index: int) -> None:  # type: ignore[override]
        try:
            le = self.lineEdit()
            if le is not None and le.hasFocus() and not self._allow_index_change:
                # Ignore implicit attempts to change index during typing
                return
        except Exception:
            pass
        try:
            super().setCurrentIndex(index)
        except Exception:
            pass

    def eventFilter(self, obj, event):  # type: ignore[override]
        try:
            from PyQt6.QtCore import QEvent
            if obj is self.lineEdit():
                if event.type() == QEvent.Type.FocusIn:
                    self._allow_index_change = False
                elif event.type() == QEvent.Type.FocusOut:
                    # Allow index change on commit
                    self._allow_index_change = True
        except Exception:
            pass
        return super().eventFilter(obj, event)


class _AccountListDelegate(QStyledItemDelegate):
    """Draws name left and token right; highlights typed substring in token."""

    def __init__(self, owner: Optional[QComboBox] = None, parent: Optional[QWidget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self._owner = owner

    def paint(self, painter, option, index):  # type: ignore[override]
        try:
            from PyQt6.QtCore import Qt as _Qt
            from PyQt6.QtGui import QFontMetrics, QPen, QPalette
            # Clone style option and clear text to avoid default text draw (prevents重なり)
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            name = str(index.data() or "")
            token = str(index.data(_Qt.ItemDataRole.UserRole + 1) or "")
            opt.text = ""
            style = opt.widget.style() if opt.widget else QApplication.style()
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

            # Compute rects for left name and right token
            rect = opt.rect
            metrics = QFontMetrics(opt.font)
            token_w = metrics.horizontalAdvance(token) + (12 if token else 0)
            name_rect = rect.adjusted(6, 0, -token_w, 0)
            token_rect = rect.adjusted(rect.width() - token_w, 0, -6, 0)

            painter.save()
            painter.setPen(opt.palette.text().color())
            painter.drawText(name_rect, _Qt.AlignmentFlag.AlignVCenter | _Qt.AlignmentFlag.AlignLeft, name)
            if token:
                # Determine query from owner lineEdit
                try:
                    import unicodedata as _ud
                    q = ""
                    if self._owner is not None and hasattr(self._owner, 'lineEdit'):
                        le = self._owner.lineEdit()
                        if le is not None:
                            q = _ud.normalize('NFKC', le.text() or "").lower().strip()
                    t_low = token.lower()
                    pos = t_low.find(q) if q else -1
                except Exception:
                    q = ""; pos = -1

                # Draw token right-aligned, highlighting match range if any
                base_pen = QPen(opt.palette.mid().color())
                acc_color = opt.palette.color(QPalette.ColorRole.Link)
                if not acc_color.isValid():
                    acc_color = QColor(30, 100, 200)
                acc_pen = QPen(acc_color)

                if pos >= 0 and q:
                    pre = token[:pos]
                    mid = token[pos:pos + len(q)]
                    post = token[pos + len(q):]
                    total_w = metrics.horizontalAdvance(token)
                    x0 = token_rect.right() - total_w
                    # vertical baseline
                    y = token_rect.y() + (token_rect.height() + metrics.ascent() - metrics.descent()) // 2
                    # draw pre
                    painter.setPen(base_pen)
                    painter.drawText(x0, y, pre)
                    # draw mid (highlight)
                    x1 = x0 + metrics.horizontalAdvance(pre)
                    painter.setPen(acc_pen)
                    painter.drawText(x1, y, mid)
                    # draw post
                    x2 = x1 + metrics.horizontalAdvance(mid)
                    painter.setPen(base_pen)
                    painter.drawText(x2, y, post)
                else:
                    painter.setPen(base_pen)
                    painter.drawText(token_rect, _Qt.AlignmentFlag.AlignVCenter | _Qt.AlignmentFlag.AlignRight, token)
            painter.restore()
        except Exception:
            super().paint(painter, option, index)


class AccountCellDelegate(QStyledItemDelegate):
    """Table delegate: shows JP text; on edit, opens QComboBox with token-aware search.

    - Paint: draw cell text and a small dropdown indicator on the right
    - Editor: QComboBox(editable) with lineEdit-attached QCompleter over [name, token]
    - Commit: writes back Japanese-only name, mapping from token if必要
    """

    def __init__(self, names: list[str], token_to_name: dict[str, str], parent: Optional[QWidget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self._names = list(names)
        # map tokens (lowercase) => JP name
        self._tok_map = {str(k).strip().lower(): v for k, v in (token_to_name or {}).items() if str(k).strip()}
        self._editing_keys: set[tuple[int,int,int]] = set()

    # ----- Painting -----
    def paint(self, painter, option, index):  # type: ignore[override]
        try:
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            # When editing this index, suppress base text to avoid double-draw
            key = (id(index.model()), index.row(), index.column())
            if key in self._editing_keys:
                opt.text = ""
            # draw default contents
            style = opt.widget.style() if opt.widget else QApplication.style()
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
            # draw dropdown indicator (small triangle) on the right
            from PyQt6.QtGui import QPen
            rect = opt.rect
            tri = "▾"
            pen = QPen(opt.palette.mid().color())
            painter.save()
            painter.setPen(pen)
            painter.drawText(rect.adjusted(rect.width()-14, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, tri)
            painter.restore()
        except Exception:
            super().paint(painter, option, index)

    # ----- Editor -----
    def createEditor(self, parent, option, index):  # type: ignore[override]
        try:
            combo = QComboBox(parent)
            combo.setEditable(True)
            try:
                combo.setCompleter(None)  # disable combobox implicit completer
            except Exception:
                pass
            # populate JP names only
            for nm in self._names:
                combo.addItem(nm)
            # attach completer to the lineEdit with [name, token]
            le = combo.lineEdit()
            if le is not None:
                # Give QObject parents to keep them alive beyond this method
                model = QStandardItemModel()
                try:
                    model.setParent(combo)
                except Exception:
                    pass
                rev: dict[str, set[str]] = {n: set() for n in self._names}
                for tok, nm in self._tok_map.items():
                    if nm in rev and tok:
                        rev[nm].add(tok)
                for nm in self._names:
                    toks = sorted(rev.get(nm, set()))
                    if toks:
                        for t in toks:
                            model.appendRow([QStandardItem(nm), QStandardItem(t)])
                    else:
                        model.appendRow([QStandardItem(nm), QStandardItem("")])
                # Parent the completer to the editor to ensure proper lifetime
                try:
                    comp = QCompleter(model, le)
                except Exception:
                    comp = QCompleter(model)
                try:
                    comp.setParent(le)
                except Exception:
                    pass
                comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                try:
                    comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
                except Exception:
                    pass
                try:
                    comp.setFilterMode(Qt.MatchFlag.MatchContains)  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    comp.setCompletionColumn(1)
                except Exception:
                    pass
                # token highlight
                try:
                    comp.popup().setItemDelegate(_AccountListDelegate(None, comp.popup()))
                except Exception:
                    pass
                try:
                    le.setCompleter(comp)
                except RuntimeError:
                    # If editor got destroyed prematurely, avoid bubbling up
                    return combo
                # Keep strong references to avoid GC collecting model/completer
                try:
                    setattr(le, "_jidou_completer", comp)
                    setattr(le, "_jidou_completer_model", model)
                except Exception:
                    pass
                # on choose, set JP name to combo
                def _on_idx(idx):
                    try:
                        nm = model.item(idx.row(), 0).text()
                        if nm:
                            combo.setCurrentText(nm)
                    except Exception:
                        pass
                try:
                    comp.activated[object].connect(_on_idx)  # type: ignore[arg-type]
                except Exception:
                    try:
                        comp.activated.connect(lambda _t: None)
                    except Exception:
                        pass
            # mark this index as editing so paint() won't draw underlying text
            try:
                key = (id(index.model()), index.row(), index.column())
                self._editing_keys.add(key)
            except Exception:
                pass
            return combo
        except Exception:
            return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):  # type: ignore[override]
        try:
            if isinstance(editor, QComboBox):
                val = str(index.model().data(index) or "")
                editor.setCurrentText(val)
                return
        except Exception:
            pass
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):  # type: ignore[override]
        try:
            if isinstance(editor, QComboBox):
                txt = (editor.currentText() or "").strip()
                # map tokens to JP if user typed token manually
                low = txt.lower()
                if low in self._tok_map:
                    txt = self._tok_map[low]
                model.setData(index, txt)
                return
        except Exception:
            pass
        super().setModelData(editor, model, index)

    def updateEditorGeometry(self, editor, option, index):  # type: ignore[override]
        try:
            editor.setGeometry(option.rect)
        except Exception:
            super().updateEditorGeometry(editor, option, index)

    def destroyEditor(self, editor, index):  # type: ignore[override]
        # remove editing flag so paint resumes normal text
        try:
            key = (id(index.model()), index.row(), index.column())
            if key in self._editing_keys:
                self._editing_keys.remove(key)
        except Exception:
            pass
        super().destroyEditor(editor, index)

class AccountLineEdit(QLineEdit):
    def __init__(self, names: list[str], token_to_name: dict[str, str]) -> None:
        super().__init__()
        self._names = list(names)
        # reverse: name -> set(tokens)
        self._rev: dict[str, set[str]] = {n: set() for n in names}
        for tok, nm in token_to_name.items():
            if nm in self._rev and tok:
                t = str(tok).strip()
                if t:
                    self._rev[nm].add(t)
        self._user_prefix: str = ""
        try:
            # Build completer model: one row per token; DisplayRole=name, UserRole+1=token
            cmodel = QStandardItemModel()
            for nm in names:
                toks = sorted(self._rev.get(nm, set()))
                if toks:
                    for t in toks:
                        it_name = QStandardItem(nm)
                        it_tok = QStandardItem(t)
                        # also stash token on UserRole+1 for delegate paint
                        it_name.setData(t, 257)  # UserRole+1
                        cmodel.appendRow([it_name, it_tok])
                else:
                    it_name = QStandardItem(nm)
                    it_tok = QStandardItem("")
                    cmodel.appendRow([it_name, it_tok])
            comp = QCompleter(cmodel)
            comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            try:
                comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            except Exception:
                pass
            try:
                comp.setFilterMode(Qt.MatchFlag.MatchContains)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                comp.setCompletionColumn(1)
            except Exception:
                pass
            # custom popup delegate to show token highlight
            try:
                popup = comp.popup()
                popup.setItemDelegate(_AccountListDelegate(None, popup))
            except Exception:
                pass
            self.setCompleter(comp)
            self._completer = comp
            self._cmodel = cmodel
            # Map completion to JP name (column 0)
            def _on_complete_index(idx):
                try:
                    nm = cmodel.item(idx.row(), 0).text()
                    if nm:
                        self.setText(nm)
                except Exception:
                    pass
            try:
                comp.activated[object].connect(_on_complete_index)  # type: ignore[arg-type]
            except Exception:
                try:
                    comp.activated.connect(lambda _t: None)
                except Exception:
                    pass
            # Normalize on focus out
            try:
                self.editingFinished.connect(lambda: self.setText(_normalize_account_text(self.text())))  # type: ignore[arg-type]
            except Exception:
                pass
        except Exception:
            pass

    def _best_candidate(self, pref: str) -> str | None:
        """Return best JP name for given ascii/jp prefix using tokens and names.
        Preference: token startswith > token contains > name startswith > name contains.
        """
        q = (pref or "").lower()
        if not q:
            return None
        # token startswith
        starts: list[str] = []
        contains: list[str] = []
        for nm, toks in self._rev.items():
            for t in toks:
                tl = t.lower()
                if tl.startswith(q):
                    starts.append(nm)
                    break
                if q in tl:
                    contains.append(nm)
                    break
        if starts:
            return starts[0]
        if contains:
            return contains[0]
        # fallback to JP name matching
        for nm in self._names:
            if nm.lower().startswith(q):
                return nm
        for nm in self._names:
            if q in nm.lower():
                return nm
        return None

    def keyPressEvent(self, ev):  # type: ignore[override]
        # Enter accepts first visible completion when popup is open; otherwise normal
        try:
            from PyQt6.QtCore import Qt as _Qt
            if ev.key() in (_Qt.Key.Key_Return, _Qt.Key.Key_Enter):
                try:
                    comp = getattr(self, '_completer', None)
                    if comp is not None and comp.popup().isVisible():
                        idx = comp.popup().currentIndex()
                        if not idx.isValid():
                            # choose first completion row
                            m = comp.completionModel()
                            row = 0
                            col = comp.completionColumn()
                            from PyQt6.QtCore import QModelIndex
                            idx = m.index(row, col) if hasattr(m, 'index') else QModelIndex()
                        # map to name (column 0) in our cmodel if possible
                        try:
                            cmodel = getattr(self, '_cmodel', None)
                            if cmodel is not None:
                                row = idx.row()
                                nm = cmodel.item(row, 0).text()
                                if nm:
                                    self.setText(nm)
                                    return
                        except Exception:
                            pass
                        # fallback: accept current text
                        self.setText(_normalize_account_text(self.text()))
                        return
                    else:
                        # popup not visible → accept current text as-is
                        self.setText(_normalize_account_text(self.text()))
                        return
                except Exception:
                    pass
        except Exception:
            pass
        return super().keyPressEvent(ev)


def _install_account_lineedits(container: QWidget, names: list[str], token_map: dict[str, str]) -> None:
    try:
        tables = container.findChildren(QTableWidget)
    except Exception:
        tables = []
    for tbl in tables:
        cols = _detect_account_columns(tbl)
        if not cols:
            continue
        try:
            rows = tbl.rowCount()
        except Exception:
            rows = 0
        for r in range(rows):
            for c in cols:
                try:
                    # Always overlay with a dropdown, even if an editor already exists
                    # (prevents row0がテキストのまま残る問題)
                    try:
                        if tbl.cellWidget(r, c) is not None:
                            tbl.removeCellWidget(r, c)
                    except Exception:
                        pass
                    it = tbl.item(r, c)
                    current = it.text().strip() if it else ""
                    # Create dropdown combo with dedicated lineEdit completer
                    combo = _AccountComboBox(names, token_map) if '_AccountComboBox' in globals() else QComboBox()
                    try:
                        combo.setEditable(True)
                        combo.setCompleter(None)  # prevent implicit combobox auto-complete
                        # Populate names (JP only)
                        for nm in names:
                            combo.addItem(nm)
                        line = combo.lineEdit()
                        if line is not None:
                            # Build completer over [name, token]
                            cmodel = QStandardItemModel()
                            rev = {n: set() for n in names}
                            for tok, nm in token_map.items():
                                if nm in rev and tok:
                                    rev[nm].add(str(tok).strip())
                            for nm in names:
                                toks = sorted(rev.get(nm, set()))
                                if toks:
                                    for t in toks:
                                        cmodel.appendRow([QStandardItem(nm), QStandardItem(t)])
                                else:
                                    cmodel.appendRow([QStandardItem(nm), QStandardItem("")])
                            comp = QCompleter(cmodel)
                            comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
                            try:
                                comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
                            except Exception:
                                pass
                            try:
                                comp.setFilterMode(Qt.MatchFlag.MatchContains)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            try:
                                comp.setCompletionColumn(1)
                            except Exception:
                                pass
                            # Beautify popup
                            try:
                                comp.popup().setItemDelegate(_AccountListDelegate(None, comp.popup()))
                            except Exception:
                                pass
                            line.setCompleter(comp)
                            # On completion, set JP name into combo (no index changes while editing)
                            def _apply_idx(idx):
                                try:
                                    row = idx.row()
                                    nm = cmodel.item(row, 0).text()
                                    if nm:
                                        combo.blockSignals(True)
                                        combo.setCurrentText(nm)
                                        combo.blockSignals(False)
                                except Exception:
                                    pass
                            try:
                                comp.activated[object].connect(_apply_idx)  # type: ignore[arg-type]
                            except Exception:
                                try:
                                    comp.activated.connect(lambda _t: None)
                                except Exception:
                                    pass
                            # While typing, keep focus and show popup; prevent index jumps
                            def _on_edit(_t: str) -> None:
                                try:
                                    if not comp.popup().isVisible():
                                        comp.complete()
                                    line.setFocus()
                                except Exception:
                                    pass
                            try:
                                line.textEdited.connect(_on_edit)  # type: ignore[arg-type]
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if current:
                        try:
                            combo.setCurrentText(current)
                        except Exception:
                            pass
                    # Put widget and clear underlying item text to avoid overlap
                    tbl.setCellWidget(r, c, combo)
                    if it is None:
                        tbl.setItem(r, c, QTableWidgetItem(""))
                    else:
                        try:
                            it.setText("")
                            it.setForeground(QBrush(QColor(0,0,0,0)))
                        except Exception:
                            pass
                except Exception:
                    continue

    def _best_map(self, txt: str) -> str:
        if not txt:
            return txt
        raw = txt.strip()
        low = raw.lower()
        # exact token match
        if raw in self._token_to_name:
            return self._token_to_name[raw]
        if low in self._token_to_name_lower:
            return self._token_to_name_lower[low]
        # heuristics: split on spaces and punctuation
        import re
        parts = [p for p in re.split(r"[^0-9A-Za-z]+", low) if p]
        # 1) if any numeric token present, match code exactly or by prefix
        codes = [p for p in parts if any(ch.isdigit() for ch in p)]
        for c in codes:
            # prefer exact code
            if c in self._token_to_name_lower:
                return self._token_to_name_lower[c]
            # or any token starting with the number
            for k, v in self._token_to_name_lower.items():
                if k.isdigit() and k.startswith(c):
                    return v
        # 2) alpha tokens: startswith match against tokens
        alphas = [p for p in parts if p.isalpha() and len(p) >= 3]
        for a in alphas:
            # prefer startswith matches of tokens
            for k, v in self._token_to_name_lower.items():
                if k.isalpha() and k.startswith(a):
                    return v
        # 3) fallback: if user typed a display name prefix, leave as-is; otherwise prefer first item
        try:
            for i in range(self.count()):
                name = self.itemText(i)
                if name and name.startswith(raw):
                    return name
        except Exception:
            pass
        return raw

    # Ensure normalization on Enter and on focus loss
    def keyPressEvent(self, ev):  # type: ignore[override]
        try:
            from PyQt6.QtCore import Qt as _Qt
            if ev.key() in ( _Qt.Key.Key_Return, _Qt.Key.Key_Enter ):
                self._apply_normalization()
        except Exception:
            pass
        return super().keyPressEvent(ev)

    def focusOutEvent(self, ev):  # type: ignore[override]
        try:
            self._apply_normalization()
        except Exception:
            pass
        return super().focusOutEvent(ev)


class _AccountDelegate(QStyledItemDelegate):
    """Table delegate that provides a dropdown with autocomplete for account columns."""

    def __init__(self, account_names: list[str], parent: Optional[QWidget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self._accounts = account_names or []

    def createEditor(self, parent, option, index):  # type: ignore[override]
        try:
            _, token_map = _load_account_catalog()
            combo = _AccountComboBox(self._accounts, token_map)
            try:
                # Show two-column-like popup using custom delegate
                combo.setItemDelegate(_AccountListDelegate(combo))
            except Exception:
                pass
            try:
                combo.currentTextChanged.connect(lambda _t: combo._apply_normalization())  # type: ignore[arg-type]
            except Exception:
                pass
            combo.setParent(parent)
            return combo
        except Exception:
            return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):  # type: ignore[override]
        try:
            val = str(index.model().data(index) or "")
            if isinstance(editor, QComboBox):
                # ensure current text matches existing cell text
                editor.setCurrentText(val)
                return
        except Exception:
            pass
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):  # type: ignore[override]
        try:
            if isinstance(editor, QComboBox):
                # normalize before saving back to model
                try:
                    txt = editor.currentText()
                    if hasattr(editor, '_best_map'):
                        txt = editor._best_map(txt)  # type: ignore[attr-defined]
                except Exception:
                    txt = editor.currentText()
                model.setData(index, txt)
                return
        except Exception:
            pass
        super().setModelData(editor, model, index)


def _detect_account_columns(tbl: 'QTableWidget') -> list[int]:  # type: ignore[name-defined]
    cols: list[int] = []
    try:
        n = tbl.columnCount()
        for c in range(n):
            try:
                header = tbl.horizontalHeaderItem(c)
                text = (header.text() if header else "") or ""
            except Exception:
                text = ""
            t = str(text)
            if not t:
                continue
            # Heuristics: contains 勘定 or 科目 and either 借方 or 貸方; exclude 金額
            if (('勘定' in t or '科目' in t) and ('借' in t or '貸' in t) and ('金額' not in t)):
                cols.append(c)
        # Fallback to common positions if nothing detected
        if not cols and n >= 4:
            cols = [1, 3]
    except Exception:
        pass
    return cols


def _install_account_cell_widgets(container: QWidget, accounts: list[str]) -> None:
    """Replace account text cells with visible comboboxes and keep item text in sync.

    Safe to call repeatedly; skips cells already using a widget.
    """
    try:
        tables = container.findChildren(QTableWidget)
    except Exception:
        tables = []
    for tbl in tables:
        cols = _detect_account_columns(tbl)
        if not cols:
            continue
        rows = 0
        try:
            rows = tbl.rowCount()
        except Exception:
            rows = 0
        for r in range(rows):
            for c in cols:
                try:
                    if tbl.cellWidget(r, c) is not None:
                        try:
                            tbl.removeCellWidget(r, c)
                        except Exception:
                            pass
                    item = tbl.item(r, c)
                    current = (item.text().strip() if item else "")
                    _, token_map = _load_account_catalog()
                    combo = _AccountComboBox(accounts, token_map)
                    try:
                        combo.setItemDelegate(_AccountListDelegate(combo))
                    except Exception:
                        pass
                    if current:
                        try:
                            combo.setCurrentText(current)
                        except Exception:
                            pass
                    # hide underlying item text to avoid visual overlap
                    try:
                        if item is not None:
                            item.setForeground(QBrush(QColor(0, 0, 0, 0)))
                    except Exception:
                        pass
                    # keep item text in sync; normalization happens on commit (editingFinished/focusOut)
                    def _sync(text, row=r, col=c, table=tbl, cb=combo):  # type: ignore[no-redef]
                        it = table.item(row, col)
                        if it is None:
                            table.setItem(row, col, QTableWidgetItem(str(text)))
                        else:
                            it.setText(str(text))
                    try:
                        combo.currentTextChanged.connect(_sync)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    tbl.setCellWidget(r, c, combo)
                except Exception:
                    continue


def _install_account_dropdowns_in_window(win: QMainWindow) -> None:  # type: ignore[name-defined]
    """Best-effort installer that scans the whole window and applies account dropdowns.

    Runs safely even if there are no matching tables.
    """
    try:
        accounts = _load_account_names()
    except Exception:
        accounts = []
    try:
        tables = win.findChildren(QTableWidget)  # type: ignore[name-defined]
    except Exception:
        tables = []
    delegate = _AccountDelegate(accounts, win)
    for tbl in tables:
        cols = _detect_account_columns(tbl)
        for c in cols:
            try:
                tbl.setItemDelegateForColumn(c, delegate)
            except Exception:
                pass
        try:
            _install_account_cell_widgets(tbl, accounts)  # type: ignore[arg-type]
        except Exception:
            pass


def _normalize_account_text(txt: str) -> str:
    """Normalize a possibly token-mixed text to the Japanese account display name.

    Uses the same token dictionary as the dropdown; safe to call anywhere.
    """
    try:
        _, token_map = _load_account_catalog()
        token_map_lower = { (k or '').strip().lower(): v for k, v in token_map.items() }
        raw = (txt or '').strip()
        if not raw:
            return raw
        low = raw.lower()
        if raw in token_map:
            return token_map[raw]
        if low in token_map_lower:
            return token_map_lower[low]
        import re
        parts = [p for p in re.split(r"[^0-9A-Za-z]+", low) if p]
        # prefer numeric exact or prefix
        for p in parts:
            if p in token_map_lower:
                return token_map_lower[p]
        for k, v in token_map_lower.items():
            for p in parts:
                if k.startswith(p):
                    return v
        return raw
    except Exception:
        return txt


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

        # 自然言語 → 会社別ルール（キーワードマッピング）
        nl_group = QGroupBox("自然言語でルール追加（会社別）")
        nlg = QVBoxLayout()
        self.nl_instr_edit = QLineEdit()
        self.nl_instr_edit.setPlaceholderText("例: 『スターバックス』は 旅費交通費/普通預金")
        nl_btn = QPushButton("指示を解析して追加")
        nl_btn.clicked.connect(self.on_nl_add)  # type: ignore[arg-type]
        nlg.addWidget(self.nl_instr_edit)
        nlg.addWidget(nl_btn)
        nl_group.setLayout(nlg)
        layout.addWidget(nl_group)

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
        try:
            llm_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        except Exception:
            pass
        try:
            # Add breathing room between rows and around the form
            llm_form.setVerticalSpacing(10)
            llm_form.setContentsMargins(8, 8, 8, 8)
        except Exception:
            pass
        # Colab/remote settings and connectivity test
        self.llm_use_colab = QLineEdit("0")
        self.llm_remote_url = QLineEdit("https://nonbeneficent-oversoftly-piper.ngrok-free.dev")
        try:
            for w in (
                self.llm_use_override,
                self.llm_provider,
                self.llm_model_path,
                self.llm_device,
                self.llm_n_gpu_layers,
                self.llm_n_threads,
                self.llm_lora_path,
                self.llm_prompt_template,
                self.llm_use_colab,
                self.llm_remote_url,
            ):
                w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                w.setMinimumWidth(280)
                w.setMinimumHeight(28)
        except Exception:
            pass
        llm_form.addRow("Override使用(1/0)", self.llm_use_override)
        llm_form.addRow("Provider", self.llm_provider)
        llm_form.addRow("Model(GGUF)", self.llm_model_path)
        llm_form.addRow("Device(cpu/gpu)", self.llm_device)
        llm_form.addRow("GPU Layers", self.llm_n_gpu_layers)
        llm_form.addRow("Threads", self.llm_n_threads)
        llm_form.addRow("LoRA Path", self.llm_lora_path)
        llm_form.addRow("Prompt Template", self.llm_prompt_template)
        llm_form.addRow("Use Colab(1/0)", self.llm_use_colab)
        _ping_row = QHBoxLayout()
        _ping_row.addWidget(self.llm_remote_url)
        _btn_ping = QPushButton("接続テスト")
        _btn_ping.clicked.connect(self.on_llm_ping)  # type: ignore[arg-type]
        try:
            _btn_ping.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            _ping_row.addWidget(_btn_ping)
            _ping_row.setStretch(0, 1)
            _ping_row.setStretch(1, 0)
        except Exception:
            _ping_row.addWidget(_btn_ping)
        _ping_container = QWidget(); _ping_container.setLayout(_ping_row)
        llm_form.addRow("Remote URL", _ping_container)
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
        # Wrap entire settings page in a scroll area to prevent vertical squeeze
        try:
            content = QWidget(); content.setLayout(layout)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(content)
            outer = QVBoxLayout(); outer.addWidget(scroll)
            self.setLayout(outer)
        except Exception:
            # Fallback: direct layout (non-scroll)
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
                QMessageBox.information(self, "保存", "LLM設定を保存しました")
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
                QMessageBox.information(self, "保存", "LLM設定を保存しました")
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
                "use_colab": (self.llm_use_colab.text().strip() == "1"),
                "remote_base_url": (self.llm_remote_url.text().strip() or None),
            }
            r = requests.post(f"{API_URL}/company_llm_settings", json=payload, timeout=10)
            if r.ok:
                QMessageBox.information(self, "保存", "LLM設定を保存しました")
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))
    def on_llm_ping(self) -> None:
        url = (
            (self.llm_remote_url.text() or "").strip()
            or "https://nonbeneficent-oversoftly-piper.ngrok-free.dev"
        )
        try:
            r = requests.post(f"{API_URL}/admin/llm_ping", json={"url": url}, timeout=5)
            if not r.ok:
                QMessageBox.warning(self, "接続テスト", f"失敗: {r.status_code} {r.text}")
                return
            data = r.json() or {}
            if data.get("ok"):
                ms = data.get("elapsed_ms")
                QMessageBox.information(self, "接続テスト", f"接続成功: {data.get('url')} ({ms} ms)")
            else:
                QMessageBox.warning(self, "接続テスト", f"接続失敗: {data.get('url')}\n{data.get('detail') or ''}")
        except Exception as e:
            QMessageBox.warning(self, "接続テスト", f"エラー: {e}")
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


    def on_nl_add(self) -> None:
        text = (self.nl_instr_edit.text() or "").strip()
        if not text:
            QMessageBox.warning(self, "入力エラー", "指示を入力してください")
            return
        try:
            r = requests.post(
                f"{API_URL}/company_nl_mapping",
                json={"company_name": self.company, "instruction": text},
                timeout=10,
            )
            if r.ok:
                QMessageBox.information(self, "保存", "LLM設定を保存しました")
                self.nl_instr_edit.setText("")
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))


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
                QMessageBox.information(self, "保存", "LLM設定を保存しました")
                self.refresh_history()
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))

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
        # スキャン進捗表示
        stat_row = QHBoxLayout()
        self.scan_status = QLabel("")
        self.scan_prog = QProgressBar(); self.scan_prog.setVisible(False)
        self.scan_prog.setTextVisible(False)
        stat_row.addWidget(self.scan_status, 1)
        stat_row.addWidget(self.scan_prog)
        layout.addLayout(stat_row)

        import_btn = QPushButton("PDF取込")
        import_dir_btn = QPushButton("フォルダ取込")
        import_btn.clicked.connect(self.import_pdfs)  # type: ignore[arg-type]
        import_dir_btn.clicked.connect(self.import_folder)  # type: ignore[arg-type]
        self.list_widget = QListWidget()

        row = QHBoxLayout()
        row.addWidget(import_btn)
        row.addWidget(import_dir_btn)
        layout.addLayout(row)
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
        if not files:
            return
        prog = QProgressDialog("取り込み中...", "中止", 0, len(files), self)
        prog.setWindowTitle("PDF取込 進捗")
        prog.setAutoClose(True); prog.setAutoReset(True)
        imported = 0
        for i, f in enumerate(files):
            try:
                with open(f, "rb") as fh:
                    files_ = {"file": (Path(f).name, fh, "application/pdf")}
                    data = {"company_name": self.company}
                    r = requests.post(f"{API_URL}/documents/import", data=data, files=files_, timeout=60)
                    if r.ok:
                        imported += 1
            except Exception as e:
                QMessageBox.warning(self, "取込失敗", f"{f}: {e}")
            prog.setValue(i + 1)
            prog.setLabelText(Path(f).name)
            if prog.wasCanceled():
                break
        # 自ページと自動仕訳ページの両方を更新
        self.refresh()
        try:
            win = self.window()
            if hasattr(win, 'review_page'):
                win.review_page.refresh()  # type: ignore[attr-defined]
        except Exception:
            pass

    def import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "フォルダ選択", self.watch_folder.text() or str(Path.cwd()))
        if not folder:
            return
        base = Path(folder)
        files = [p for p in base.rglob("*.pdf")]
        if not files:
            QMessageBox.information(self, "保存", "LLM設定を保存しました")
            return
        prog = QProgressDialog("フォルダ取込中...", "中止", 0, len(files), self)
        prog.setWindowTitle("フォルダ取込 進捗")
        prog.setAutoClose(True); prog.setAutoReset(True)
        imported = 0
        for i, p in enumerate(files):
            try:
                with open(p, "rb") as fh:
                    files_ = {"file": (p.name, fh, "application/pdf")}
                    data = {"company_name": self.company}
                    r = requests.post(f"{API_URL}/documents/import", data=data, files=files_, timeout=60)
                    if r.ok:
                        imported += 1
            except Exception:
                pass
            prog.setValue(i + 1)
            try:
                prog.setLabelText(f"{i+1}/{len(files)}: {p.name}")
            except Exception:
                pass
            if prog.wasCanceled():
                break
        self.refresh()
        try:
            win = self.window()
            if hasattr(win, 'review_page'):
                win.review_page.refresh()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self.scan_status.setText(f"フォルダ取込 完了: {imported} 件")
        except Exception:
            pass

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
            QMessageBox.warning(self, "保存失敗", str(e))

        if not ok:
            QMessageBox.warning(self, "スキャン開始失敗", "ScanSnap Homeを起動できませんでした。")
            self.scan_btn.setEnabled(True)
            return

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_folder)  # type: ignore[arg-type]
        self._poll_deadline = time.time() + 120
        self._poll_timer.start(1000)
        # 進捗UI
        self.scan_status.setText("スキャン待機中...（最大120秒）")
        self.scan_prog.setVisible(True)
        self.scan_prog.setRange(0, 0)  # 不確定バー
        self._import_count = 0

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
            # スキャンで新規取り込みがあれば両ページ更新
            self.refresh()
            try:
                win = self.window()
                if hasattr(win, 'review_page'):
                    win.review_page.refresh()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                self._import_count += 1
                self.scan_status.setText(f"スキャン取り込み中... {self._import_count} 件")
            except Exception:
                pass
        if time.time() > getattr(self, "_poll_deadline", 0.0):
            if getattr(self, "_poll_timer", None):
                self._poll_timer.stop()
                self._poll_timer = None
            self.scan_btn.setEnabled(True)
            try:
                self.scan_prog.setVisible(False)
                self.scan_status.setText(f"スキャン完了: {getattr(self, '_import_count', 0)} 件")
            except Exception:
                pass


class ReviewPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QHBoxLayout()

        # Left: PDF preview + NL instruction
        left = QVBoxLayout()
        # Keep a reference for spacing/margins in sizing logic
        self._left_layout = left
        self.pdf_label = QLabel("PDFプレビュー")
        # 固定幅の表示枠を確保して、ページ移動でレイアウトが揺れないようにする
        try:
            self.pdf_label.setMinimumWidth(int(UI_SETTINGS.get("pdf_label_min_width", 420)))
            maxw = int(UI_SETTINGS.get("pdf_label_max_width", 0))
            if maxw and maxw > 0:
                self.pdf_label.setMaximumWidth(maxw)
        except Exception:
            self.pdf_label.setMinimumWidth(420)
        self.pdf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pdf_label.setStyleSheet("border:1px solid #ccc; background:white;")
        # 横方向にしっかり広がるようにする
        try:
            self.pdf_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        except Exception:
            pass
        # 以降のスケーリングは設定値を基準に行う
        self._pdf_fixed_width = int(UI_SETTINGS.get("pdf_fixed_width", 420))
        self._pdf_scale_ratio = float(UI_SETTINGS.get("pdf_scale_ratio", 1.25))
        left.addWidget(self.pdf_label, 3)

        nl_group = QGroupBox("自然言語の指示（修正/仕訳）")
        try:
            nl_group.setMinimumHeight(int(UI_SETTINGS.get("nl_group_min_height", 60)))
            m = int(UI_SETTINGS.get("nl_group_max_height", 0))
            if m and m > 0:
                nl_group.setMaximumHeight(m)
        except Exception:
            pass
        nlg = QHBoxLayout()
        self.nl_edit = QTextEdit()
        # 明示的に編集可能にする（フォーカスも有効化）
        try:
            self.nl_edit.setEnabled(True)
            self.nl_edit.setReadOnly(False)
            self.nl_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        except Exception:
            pass
        self.nl_edit.setPlaceholderText("例: 『スタバ』は 旅費交通費/現金")
        # 折り返しを有効にし、高さを確保
        try:
            from PyQt6.QtGui import QTextOption  # type: ignore
            self.nl_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        except Exception:
            pass
        try:
            self.nl_edit.setMinimumHeight(int(UI_SETTINGS.get("nl_edit_min_height", 72)))
            mx = int(UI_SETTINGS.get("nl_edit_max_height", 0))
            if mx and mx > 0:
                self.nl_edit.setMaximumHeight(mx)
        except Exception:
            self.nl_edit.setMinimumHeight(72)
        try:
            self.nl_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        except Exception:
            pass
        nl_btn = QPushButton("適用/学習")
        nl_btn.clicked.connect(self._apply_nl)  # type: ignore[arg-type]
        nlg.addWidget(self.nl_edit)
        nlg.addWidget(nl_btn)
        nl_group.setLayout(nlg)
        # 後段の高さ計算で参照できるように保持
        self.nl_group = nl_group
        left.addWidget(nl_group, 1)

        # 左ペインはスクロール可能にして、全画面やタスクバーで高さが圧迫されても
        # 下部の自然言語入力欄が隠れないようにする
        left_container = QWidget(); left_container.setLayout(left)
        # 左ペインが極端に潰れないよう下限を持たせる
        try:
            left_container.setMinimumWidth(int(UI_SETTINGS.get("left_container_min_width", 500)))
            mx = int(UI_SETTINGS.get("left_container_max_width", 0))
            if mx and mx > 0:
                left_container.setMaximumWidth(mx)
        except Exception:
            pass
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        try:
            left_scroll.setMinimumWidth(int(UI_SETTINGS.get("left_scroll_min_width", 480)))
            mx = int(UI_SETTINGS.get("left_scroll_max_width", 0))
            if mx and mx > 0:
                left_scroll.setMaximumWidth(mx)
        except Exception:
            pass
        try:
            left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        except Exception:
            pass
        # ビューポートのリサイズに追従してPDFのスケールを調整
        try:
            left_scroll.viewport().installEventFilter(self)
        except Exception:
            pass

        # Right: Journal-style grid (2 rows per entry) with inline edit
        # Row 1: 日付 / 伝票No. / 借方勘定科目 / 借方金額 / 貸方勘定科目 / 貸方金額 / 摘要 / 請求書区分
        # Row 2:           ""     / 借方補助科目   /            / 貸方補助科目   /            / 税区分
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "日付\n",
            "借方勘定科目\n（補助科目）",
            "借方金額\n",
            "貸方勘定科目\n（補助科目）",
            "貸方金額\n",
            "摘要\n（税区分）",
            "請求書区分\n",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        try:
            # make header tall enough to show two lines
            self.table.horizontalHeader().setFixedHeight(42)
            self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        except Exception:
            pass
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # Enable inline editing
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.SelectedClicked | QTableWidget.EditTrigger.EditKeyPressed
        )
        self.table.itemSelectionChanged.connect(self._on_table_select)  # type: ignore[arg-type]
        self.table.itemChanged.connect(self._on_item_changed)  # type: ignore[arg-type]
        # Install delegates and visible dropdowns directly on this table (debit=1, credit=3)
        try:
            names, token_map = _load_account_catalog()
            if names:
                delegate = AccountCellDelegate(names, token_map, self.table)
                try:
                    self.table.setItemDelegateForColumn(1, delegate)
                    self.table.setItemDelegateForColumn(3, delegate)
                except Exception:
                    pass
        except Exception:
            pass
        # Toolbar for save/later/delete
        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("保存/OK（学習）")
        self.btn_later = QPushButton("後で確認")
        self.btn_delete = QPushButton("削除")
        self.btn_save.clicked.connect(self._save_selected)  # type: ignore[arg-type]
        self.btn_later.clicked.connect(self._mark_later)  # type: ignore[arg-type]
        self.btn_delete.clicked.connect(self._delete_selected)  # type: ignore[arg-type]
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_later)
        btn_row.addWidget(self.btn_delete)

        # Tax category quick apply and PDF pager
        util_row = QHBoxLayout()
        self.tax_combo = QComboBox(); self._load_tax_categories()
        apply_tax = QPushButton("税区分適用")
        apply_tax.clicked.connect(self._apply_tax_combo)  # type: ignore[arg-type]
        self.page_prev = QPushButton("＜")
        self.page_next = QPushButton("＞")
        self.page_label = QLabel("1/1")
        self.page_prev.clicked.connect(lambda: self._change_page(-1))  # type: ignore[arg-type]
        self.page_next.clicked.connect(lambda: self._change_page(1))  # type: ignore[arg-type]
        # Intra-PDF page navigation
        self.pdf_prev = QPushButton("＜頁")
        self.pdf_next = QPushButton("頁＞")
        self.pdf_page_label = QLabel("0/0")
        self.pdf_prev.clicked.connect(lambda: self._change_pdf_page(-1))  # type: ignore[arg-type]
        self.pdf_next.clicked.connect(lambda: self._change_pdf_page(1))  # type: ignore[arg-type]
        util_row.addWidget(QLabel("税区分:"))
        util_row.addWidget(self.tax_combo)
        util_row.addWidget(apply_tax)
        util_row.addStretch(1)
        util_row.addWidget(self.pdf_prev)
        util_row.addWidget(self.pdf_page_label)
        util_row.addWidget(self.pdf_next)
        util_row.addWidget(self.page_prev)
        util_row.addWidget(self.page_label)
        util_row.addWidget(self.page_next)

        right = QVBoxLayout()
        right.addLayout(btn_row)
        right.addLayout(util_row)
        right.addWidget(self.table)
        right_wrap = QWidget(); right_wrap.setLayout(right)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        # 全画面時に左ペインが潰れてNL入力欄が見えなくなるのを防止
        try:
            splitter.setChildrenCollapsible(False)
            # Guard: avoid calling setCollapsible before widgets are added
            if splitter.count() >= 1:
                splitter.setCollapsible(0, False)
            if splitter.count() >= 2:
                splitter.setCollapsible(1, False)
        except Exception:
            pass
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_wrap)
        # After widgets are added, enforce non-collapsible explicitly (safe)
        try:
            if splitter.count() >= 1:
                splitter.setCollapsible(0, False)
            if splitter.count() >= 2:
                splitter.setCollapsible(1, False)
        except Exception:
            pass
        try:
            splitter.setStretchFactor(0, int(UI_SETTINGS.get("splitter_stretch_left", 3)))
            splitter.setStretchFactor(1, int(UI_SETTINGS.get("splitter_stretch_right", 4)))
        except Exception:
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)
        # 初期割り当てを左を1.2倍広くする
        self._splitter = splitter
        # Keep left width stable on window resizes if fixed policy is set
        try:
            splitter.splitterMoved.connect(lambda pos, idx: self._apply_splitter_policy())  # type: ignore[arg-type]
            self.installEventFilter(self)
        except Exception:
            pass
        try:
            from PyQt6.QtCore import QTimer
            def _boost_left():
                self._apply_splitter_policy()
            QTimer.singleShot(0, _boost_left)
        except Exception:
            pass

        # 旧・単票入力フォームと左リストは廃止。
        # 以降は上部のジャーナル表で一括表示・編集・保存します。
        self.setLayout(layout)
        # Load initial grid
        self._load_unconfirmed()

        self.refresh()

    def eventFilter(self, obj, event):  # type: ignore[override]
        try:
            from PyQt6.QtCore import QEvent
            et = event.type()
            if et in (QEvent.Type.Resize, QEvent.Type.LayoutRequest):
                self._apply_splitter_policy()
                # 左側ビューポートのリサイズ時はPDFスケールも追従
                try:
                    from PyQt6.QtWidgets import QScrollArea
                    if isinstance(obj, QScrollArea) or getattr(obj, 'objectName', lambda: '')() == 'qt_scrollarea_viewport':
                        if getattr(self, '_pdf_path', None):
                            self._show_pdf(getattr(self, '_pdf_path'))
                except Exception:
                    pass
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _apply_splitter_policy(self) -> None:
        try:
            sizes = self._splitter.sizes()
            if len(sizes) < 2:
                return
            total = sum(sizes) if sum(sizes) > 0 else max(1, self._splitter.width())
            # Apply only when maximized if requested
            try:
                only_on_max = bool(UI_SETTINGS.get("apply_splitter_only_when_maximized", False))
            except Exception:
                only_on_max = False
            if only_on_max:
                try:
                    win = self.window()
                    if hasattr(win, 'isMaximized') and not win.isMaximized():  # type: ignore[attr-defined]
                        return
                except Exception:
                    pass
            # 設定: 左の最小/最大(px, 比率)
            min_left = int(UI_SETTINGS.get("splitter_min_left_px", 560))
            ratio = float(UI_SETTINGS.get("splitter_left_ratio", 0.35))
            rratio = float(UI_SETTINGS.get("splitter_right_ratio", 0.0))
            # If right ratio is specified (including 1.0), honor it and compute left from it
            if rratio > 0.0:
                rratio = max(0.0, min(1.0, rratio))
                ratio = max(0.0, min(1.0, 1.0 - rratio))
            # If left ratio is negative, treat as 'ignore' and rely on right ratio / fixed_px
            if ratio < 0.0:
                ratio = 0.0
            max_left_px = int(UI_SETTINGS.get("splitter_max_left_px", 0))
            max_ratio = float(UI_SETTINGS.get("splitter_left_max_ratio", 1.0))
            fixed_px = int(UI_SETTINGS.get("splitter_left_fixed_px", 0))
            fixed_ratio = float(UI_SETTINGS.get("splitter_left_fixed_ratio", 0.0))
            if fixed_px and fixed_px > 0:
                requested = fixed_px
            elif 0.0 < fixed_ratio < 1.0:
                requested = int(total * fixed_ratio)
            else:
                requested = max(min_left, int(total * ratio))
            lmin_ratio = float(UI_SETTINGS.get("splitter_left_min_ratio", 0.0))
            if 0.0 < lmin_ratio < 1.0:
                requested = max(requested, int(total * lmin_ratio))
            if max_left_px and max_left_px > 0:
                requested = min(requested, max_left_px)
            if 0 < max_ratio <= 1.0:
                requested = min(requested, int(total * max_ratio))
            left = min(total - 1, max(1, requested))
            right = max(1, total - left)
            self._splitter.setSizes([left, right])
        except Exception:
            pass

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

    # Load tax categories once
    def _load_tax_categories(self) -> None:
        try:
            r = requests.get(f"{API_URL}/tax_categories", timeout=10)
            if r.ok:
                self._tax_map = {}
                self.tax_combo.clear()
                for row in r.json():
                    name = row.get("name") or ""
                    code = row.get("code") or name
                    self.tax_combo.addItem(name, code)
                    self._tax_map[code] = name
        except Exception:
            pass

    def _load_document(self, d: dict) -> None:
        # Fill edit form from a document dict
        try:
            self.current_doc_id = int(d.get("id"))
        except Exception:
            self.current_doc_id = None
        auto = d.get("auto") or {}
        # date
        try:
            from PyQt6.QtCore import QDate
            dt = (auto.get("date") or "").strip()
            if dt and len(dt.split("/")) == 3:
                y, m, da = [int(x) for x in dt.split("/")]
                self.date.setDate(QDate(y, m, da))
            else:
                self.date.setDate(QDate.currentDate())
        except Exception:
            try:
                from PyQt6.QtCore import QDate
                self.date.setDate(QDate.currentDate())
            except Exception:
                pass
        # basics
        self.amount.setText(str(auto.get("amount") or ""))
        self.summary.setEditText(auto.get("summary") or "")
        self.debit.setText(auto.get("debit_account") or "")
        self.credit.setText(auto.get("credit_account") or "")
        # populate and set subaccounts and invoice
        try:
            self._populate_from_accounts()
        except Exception:
            pass
        self.debit_sub.setCurrentText(auto.get("debit_subaccount") or "")
        self.credit_sub.setCurrentText(auto.get("credit_subaccount") or "")
        self.invoice.setCurrentText(auto.get("invoice_status") or "")

    # -----------------------
    # Journal-style grid helpers (per-PDF rendering)
    # -----------------------
    def _load_unconfirmed(self) -> None:
        try:
            self._loading = True
            # fetch account maps once
            self._fetch_account_maps()
            r = requests.get(
                f"{API_URL}/documents",
                params={"company_name": self.company, "status": "unconfirmed"},
                timeout=10,
            )
            if not r.ok:
                self._docs = []
                self._doc_index = 0
                self._render_current_doc()
                return
            self._docs = list(r.json() or [])
            # Keep current index if possible; otherwise clamp
            curr = getattr(self, "_doc_index", 0)
            if not self._docs:
                self._doc_index = 0
            else:
                self._doc_index = max(0, min(curr, len(self._docs) - 1))
            self._render_current_doc()
        except Exception:
            try:
                self._docs = []
                self._doc_index = 0
                self._render_current_doc()
            except Exception:
                pass
        finally:
            self._loading = False

    def _render_current_doc(self) -> None:
        # Update label with current doc position
        total = len(getattr(self, "_docs", []))
        idx = getattr(self, "_doc_index", 0)
        try:
            self.page_label.setText(f"{(idx+1) if total else 0}/{total}")
        except Exception:
            pass

        # Clear table if no documents
        if total == 0:
            self.table.setRowCount(0)
            self._row_to_id = []
            self._show_pdf(None)
            return

        d = self._docs[idx]
        auto = d.get("auto") or {}
        date = auto.get("date") or ""
        amount = abs(auto.get("amount") or 0)
        debit = auto.get("debit_account") or ""
        credit = auto.get("credit_account") or ""
        debit_sub = auto.get("debit_subaccount") or ""
        credit_sub = auto.get("credit_subaccount") or ""
        summary = auto.get("summary") or ""
        tax = auto.get("invoice_status") or ""

        # Set up a 2-row table for this single PDF's entry
        self.table.setRowCount(2)
        self._row_to_id = [int(d.get("id")), int(d.get("id"))]

        r0 = 0
        r1 = 1
        vals_r1 = [
            date,
            debit,
            f"{amount:,}",
            credit,
            f"{amount:,}",
            summary,
            tax,
        ]
        for j, v in enumerate(vals_r1):
            if j in (5, 6):
                continue
            self.table.setItem(r0, j, QTableWidgetItem(str(v)))

        vals_r2 = [
            "",
            debit_sub,
            "",
            credit_sub,
            "",
            auto.get("invoice_status") or "",
            "",
        ]
        for j, v in enumerate(vals_r2):
            if j in (1, 3, 5):
                continue
            self.table.setItem(r1, j, QTableWidgetItem(str(v)))

        self._setup_row_widgets(r0, r1, debit, credit, debit_sub, credit_sub, summary, tax)

        # Show the current document's PDF (first page by default)
        try:
            self._pdf_page = 0
            self._show_pdf(d.get("file_path"))
        except Exception:
            pass

    def _on_table_select(self) -> None:
        sel = self.table.selectedIndexes()
        if not sel:
            return
        row = sel[0].row()
        try:
            doc_id = self._row_to_id[row]
        except Exception:
            return
        # テーブル編集に一本化したため、PDFプレビューのみ更新
        try:
            for d in self._docs or []:
                if d.get("id") == doc_id:
                    try:
                        self._show_pdf(d.get("file_path"))
                    except Exception:
                        pass
                    break
        except Exception:
            pass

    def _fetch_account_maps(self) -> None:
        # Load account subaccounts and summaries for this company
        try:
            r = requests.get(f"{API_URL}/account_settings", params={"company_name": self.company}, timeout=10)
            subs_map: dict[str, list[str]] = {}
            sums_map: dict[str, list[str]] = {}
            if r.ok:
                for row in r.json() or []:
                    acc = row.get("account_name") or ""
                    subs_map[acc] = list(row.get("subaccounts") or [])
                    sums_map[acc] = list(row.get("summaries") or [])
            self._acct_subs = subs_map
            self._acct_summaries = sums_map
        except Exception:
            self._acct_subs = {}
            self._acct_summaries = {}

    def _setup_row_widgets(self, r0: int, r1: int, debit: str, credit: str, debit_sub: str, credit_sub: str, summary: str, tax_name: str) -> None:
        # Summary combo (row0 col5)
        # clear any text items to avoid double rendering below widgets
        try:
            self.table.takeItem(r0, 5)
            self.table.takeItem(r0, 6)
            self.table.takeItem(r1, 1)
            self.table.takeItem(r1, 3)
            self.table.takeItem(r1, 5)
        except Exception:
            pass
        sum_cb = QComboBox(); sum_cb.setEditable(True)
        opts = [""] + list(self._acct_summaries.get(debit, []))
        sum_cb.addItems(opts)
        if summary:
            sum_cb.setCurrentText(summary)
        self.table.setCellWidget(r0, 5, sum_cb)

        # Invoice combo (row0 col6)
        inv_cb = QComboBox(); inv_cb.addItems(["", "適格", "非適格", "非課税"])
        inv_cb.setCurrentText(tax_name or "")
        self.table.setCellWidget(r0, 6, inv_cb)

        # Debit sub (row1 col1)
        dsub = QComboBox(); dsub.setEditable(True)
        dsub.addItems([""] + list(self._acct_subs.get(debit, [])))
        dsub.setCurrentText(debit_sub or "")
        self.table.setCellWidget(r1, 1, dsub)
        # Credit sub (row1 col3)
        csub = QComboBox(); csub.setEditable(True)
        csub.addItems([""] + list(self._acct_subs.get(credit, [])))
        csub.setCurrentText(credit_sub or "")
        self.table.setCellWidget(r1, 3, csub)
        # Tax category (row1 col5)
        tcb = QComboBox();
        try:
            # fill from tax map if available
            for code, name in getattr(self, "_tax_map", {}).items():
                tcb.addItem(name, code)
        except Exception:
            pass
        tcb.setEditable(False)
        tcb.setCurrentText(tax_name or "")
        self.table.setCellWidget(r1, 5, tcb)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if getattr(self, "_loading", False):
            return
        r = item.row()
        c = item.column()
        # If tax cell (second row, column 5) received a text item, sync to widget and remove item
        if r % 2 == 1 and c == 5:
            try:
                w = self.table.cellWidget(r, 5)
                txt = item.text().strip()
                # QComboBox is imported at module level; avoid local import to prevent scope issues
                if isinstance(w, QComboBox) and txt:
                    w.setCurrentText(txt)
                # remove overlay item to avoid double rendering
                self.table.takeItem(r, 5)
            except Exception:
                pass
            return
        # Only respond for top rows and debit/credit edits
        if r % 2 == 0 and c in (1, 3):
            r0 = r
            r1 = r + 1
            debit = self.table.item(r0, 1).text().strip() if self.table.item(r0, 1) else ""
            credit = self.table.item(r0, 3).text().strip() if self.table.item(r0, 3) else ""
            # Update subaccount combos and summary options
            dsub: QComboBox = self.table.cellWidget(r1, 1)  # type: ignore[assignment]
            csub: QComboBox = self.table.cellWidget(r1, 3)  # type: ignore[assignment]
            sum_cb: QComboBox = self.table.cellWidget(r0, 5)  # type: ignore[assignment]
            if isinstance(dsub, QComboBox):
                dsub.clear(); dsub.addItems([""] + list(getattr(self, "_acct_subs", {}).get(debit, [])))
            if isinstance(csub, QComboBox):
                csub.clear(); csub.addItems([""] + list(getattr(self, "_acct_subs", {}).get(credit, [])))
            if isinstance(sum_cb, QComboBox):
                sum_cb.clear(); sum_cb.addItems([""] + list(getattr(self, "_acct_summaries", {}).get(debit, [])))

    # PDF preview helpers
    def _show_pdf(self, path: Optional[str]) -> None:
        self._pdf_path = path or None
        self._pdf_page = getattr(self, '_pdf_page', 0)
        if not path:
            # 画像が無い場合もラベルの最小サイズを維持してレイアウトが崩れないようにする
            self.pdf_label.setText("PDFなし")
            return
        try:
            try:
                import fitz  # PyMuPDF
            except Exception as e:
                # Try automatic install if missing
                if 'No module named' in str(e):
                    try:
                        import subprocess, sys
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pymupdf>=1.23,<2'])
                        import fitz  # type: ignore
                    except Exception:
                        raise
                else:
                    raise
            doc = fitz.open(path)
            # clamp page
            if self._pdf_page < 0:
                self._pdf_page = 0
            if self._pdf_page >= len(doc):
                self._pdf_page = max(0, len(doc)-1)
            page = doc.load_page(self._pdf_page)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            # PyQt6 は列挙名が Format_RGBA8888 / Format_RGB888
            fmt = QImage.Format.Format_RGBA8888 if getattr(pix, 'alpha', False) else QImage.Format.Format_RGB888
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
            pm = QPixmap.fromImage(img)
            # ビューポート幅×倍率(既定1.25)を狙いつつ、縦スクロールが出ないよう高さ制約も反映
            try:
                vp = None
                # 直近で作ったスクロールを検索
                p = self.pdf_label.parent()
                while p is not None and not isinstance(p, QScrollArea):
                    p = p.parent()
                if isinstance(p, QScrollArea):
                    vp = p.viewport()
                avail_w = vp.width() if vp is not None else self.pdf_label.width()
                avail_h = vp.height() if vp is not None else self.pdf_label.height()
            except Exception:
                avail_w = self.pdf_label.width()
                avail_h = self.pdf_label.height()
            try:
                ratio = float(getattr(self, "_pdf_scale_ratio", 1.25))
                # 設定による下限/上限
                rmin = float(UI_SETTINGS.get("pdf_scale_ratio_min", 0.0))
                rmax = float(UI_SETTINGS.get("pdf_scale_ratio_max", 0.0))
                if rmin:
                    ratio = max(rmin, ratio)
                if rmax:
                    ratio = min(rmax, ratio)
            except Exception:
                ratio = 1.25
            # 希望倍率(1.25)を優先しつつ、ビューポート幅の98%を上限にして横スクロールを回避
            fixed_min = getattr(self, "_pdf_fixed_width", 420)
            desired_w = int(avail_w * ratio)
            max_w_by_width = int(avail_w * 0.98)
            # 縦方向の制約: NL入力欄の必要高さを差し引いたPDF許容高さを計算
            try:
                nl_h = int(self.nl_group.sizeHint().height()) if hasattr(self, 'nl_group') else 0
            except Exception:
                nl_h = 0
            # レイアウトのマージンとスペーシングも考慮して、縦スクロールが出ないように余白を差し引く
            extra = 16
            try:
                left_layout = getattr(self, "_left_layout", None)
                if left_layout is not None:
                    m = left_layout.contentsMargins()
                    extra += int(m.top()) + int(m.bottom()) + int(left_layout.spacing() or 0)
            except Exception:
                pass
            try:
                reserve = int(UI_SETTINGS.get("nl_reserved_height_px", 0))
            except Exception:
                reserve = 0
            reserve = max(reserve, nl_h)
            allowed_h = max(200, avail_h - reserve - extra)
            # ピクセルから高さ基準の最大幅を算出（アスペクト比維持）
            try:
                max_w_by_height = int(pm.width() * (allowed_h / max(1, pm.height())))
            except Exception:
                max_w_by_height = max_w_by_width
            hard_max_w = max(100, min(max_w_by_width, max_w_by_height))
            # 追加の上限（絶対px）
            try:
                max_px = int(UI_SETTINGS.get("pdf_max_width_px", 0))
                if max_px and max_px > 0:
                    hard_max_w = min(hard_max_w, max_px)
            except Exception:
                pass
            target = int(max(fixed_min, min(desired_w, hard_max_w)))
            w = max(320, target)
            # 高さの上限に確実に収めるため、幅と高さの両方を指定したスケールを用いる
            try:
                from PyQt6.QtCore import Qt as _Qt
                pm = pm.scaled(w, int(allowed_h), _Qt.AspectRatioMode.KeepAspectRatio, _Qt.TransformationMode.SmoothTransformation)
            except Exception:
                pm = pm.scaledToWidth(w, Qt.TransformationMode.SmoothTransformation)
            # ラベル自体の高さ上限も設定して、NL欄が初期から見えるようにする
            try:
                self.pdf_label.setMaximumHeight(int(allowed_h))
            except Exception:
                pass
            self.pdf_label.setPixmap(pm)
            # Update intra-PDF page position label (keep doc navigation label separate)
            try:
                self.pdf_page_label.setText(f"{self._pdf_page+1}/{len(doc)}")
            except Exception:
                pass
            doc.close()
        except Exception as e:
            self.pdf_label.setText(f"PDF表示エラー\n{e}")

    def _change_page(self, delta: int) -> None:
        # Repurposed: use '<' and '>' to move across PDFs
        total = len(getattr(self, "_docs", []))
        if total == 0:
            return
        idx = getattr(self, "_doc_index", 0) + delta
        if idx < 0:
            idx = 0
        if idx >= total:
            idx = total - 1
        self._doc_index = idx
        self._render_current_doc()

    def _change_pdf_page(self, delta: int) -> None:
        # Navigate within current PDF pages
        if not getattr(self, '_pdf_path', None):
            return
        try:
            self._pdf_page = max(0, (getattr(self, '_pdf_page', 0) + delta))
            self._show_pdf(self._pdf_path)
        except Exception:
            pass

    # ビューポート/ウィンドウのリサイズでプレビューを再スケール
    def eventFilter(self, obj, event):  # type: ignore[override]
        try:
            from PyQt6.QtCore import QEvent
            if event.type() in (QEvent.Type.Resize,):
                if getattr(self, "_pdf_path", None):
                    self._show_pdf(self._pdf_path)
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _apply_tax_combo(self) -> None:
        # apply selected tax category display name to table second row tax cell
        sel = self.table.selectedIndexes()
        if not sel:
            return
        r = sel[0].row()
        r1 = r - (r % 2) + 1
        code = self.tax_combo.currentData()
        name = self.tax_combo.currentText()
        item = QTableWidgetItem(f"税区分: {name}")
        item.setForeground(QBrush(QColor(90,90,90)))
        # tax cell column index 5 in our 7-column layout
        self.table.setItem(r1, 5, item)
        # 直後にウィジェットへ反映して Item を除去（二重描画防止）
        try:
            w = self.table.cellWidget(r1, 5)
            if isinstance(w, QComboBox):
                if name:
                    w.setCurrentText(name)
                self.table.takeItem(r1, 5)
        except Exception:
            pass

    # Save/later/delete
    def _selected_doc_id_and_rows(self) -> tuple[Optional[int], Optional[int], Optional[int]]:
        sel = self.table.selectedIndexes()
        if not sel:
            return None, None, None
        r = sel[0].row()
        # Normalize to top row of pair
        r0 = r - (r % 2)
        r1 = r0 + 1
        doc_id = self._row_to_id[r0]
        return doc_id, r0, r1

    def _collect_from_table(self, r0: int, r1: int) -> dict:
        # Extract values from pair rows
        def val(r, c):
            w = self.table.cellWidget(r, c)
            if isinstance(w, QComboBox):
                return w.currentText().strip()
            try:
                from PyQt6.QtWidgets import QLineEdit as _QLineEdit
                if isinstance(w, _QLineEdit):
                    return w.text().strip()
            except Exception:
                pass
            it = self.table.item(r, c)
            return it.text().strip() if it else ""

        date = val(r0, 0)
        debit = _normalize_account_text(val(r0, 1))
        debit_amt = val(r0, 2)
        credit = _normalize_account_text(val(r0, 3))
        credit_amt = val(r0, 4)
        summary = val(r0, 5)
        invoice = val(r0, 6)
        # Second row
        debit_sub = val(r1, 1)
        credit_sub = val(r1, 3)
        tax_text = val(r1, 5)

        # Prefer debit amount unless empty, else credit
        try:
            amount = int(str(debit_amt).replace(",", "")) if debit_amt else int(str(credit_amt).replace(",", ""))
        except Exception:
            amount = None
        return {
            "date": date or None,
            "amount": amount,
            "summary": summary or None,
            "debit": debit or None,
            "credit": credit or None,
            "debit_sub": debit_sub or None,
            "credit_sub": credit_sub or None,
            "invoice_status": invoice or (tax_text or None),
        }

    def _save_selected(self) -> None:
        doc_id, r0, r1 = self._selected_doc_id_and_rows()
        if doc_id is None or r0 is None or r1 is None:
            QMessageBox.information(self, "保存", "LLM設定を保存しました")
            return
        data = self._collect_from_table(r0, r1)
        try:
            payload = {
                "date": data["date"],
                "amount": data["amount"],
                "summary": data["summary"],
                "debit": data["debit"],
                "credit": data["credit"],
                "debit_sub": data["debit_sub"],
                "credit_sub": data["credit_sub"],
                "invoice_status": data["invoice_status"],
            }
            r = requests.post(
                f"{API_URL}/documents/{doc_id}/ok",
                json={
                    "date": payload.get("date"),
                    "amount": payload.get("amount"),
                    "summary": payload.get("summary"),
                    "debit_account": payload.get("debit"),
                    "credit_account": payload.get("credit"),
                    "debit_subaccount": payload.get("debit_sub"),
                    "credit_subaccount": payload.get("credit_sub"),
                    "invoice_status": payload.get("invoice_status"),
                },
                timeout=20,
            )
            if r.ok:
                QMessageBox.information(self, "保存", "LLM設定を保存しました")
                self._load_unconfirmed()
                try:
                    win = self.window()
                    if hasattr(win, 'scan_page'):
                        win.scan_page.refresh()  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))

    def _mark_later(self) -> None:
        doc_id, _, _ = self._selected_doc_id_and_rows()
        if doc_id is None:
            return
        try:
            requests.post(f"{API_URL}/documents/{doc_id}/check", timeout=10)
            self._load_unconfirmed()
            try:
                win = self.window()
                if hasattr(win, 'scan_page'):
                    win.scan_page.refresh()  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

    def _delete_selected(self) -> None:
        # Prefer table selection; if none, use current doc in view
        doc_id, _, _ = self._selected_doc_id_and_rows()
        if doc_id is None:
            try:
                idx = int(getattr(self, "_doc_index", 0))
                docs = list(getattr(self, "_docs", []) or [])
                if 0 <= idx < len(docs):
                    doc_id = int(docs[idx].get("id"))
            except Exception:
                doc_id = None
        if not doc_id:
            return
        # Confirm delete
        try:
            resp = QMessageBox.question(
                self,
                "削除の確認",
                "この仕訳を削除しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        except Exception:
            pass
        # Delete via API and refresh
        try:
            requests.delete(f"{API_URL}/documents/{doc_id}", timeout=10)
        except Exception:
            pass
        try:
            # Keep index in range after deletion
            curr = int(getattr(self, "_doc_index", 0))
            total = max(0, len(getattr(self, "_docs", []) or []) - 1)
            setattr(self, "_doc_index", max(0, min(curr, total)))
        except Exception:
            pass
        try:
            self._load_unconfirmed()
            win = self.window()
            if hasattr(win, 'scan_page'):
                win.scan_page.refresh()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _apply_nl(self) -> None:
        # QTextEdit への変更に伴い toPlainText で取得
        try:
            text = (self.nl_edit.toPlainText() or "").strip()
        except Exception:
            text = (self.nl_edit.text() or "").strip()
        if not text:
            return
        try:
            r = requests.post(
                f"{API_URL}/company_nl_mapping",
                json={"company_name": self.company, "instruction": text},
                timeout=15,
            )
            if r.ok:
                QMessageBox.information(self, "保存", "LLM設定を保存しました")
                try:
                    self.nl_edit.setPlainText("")
                except Exception:
                    self.nl_edit.setText("")
            else:
                QMessageBox.warning(self, "保存失敗", r.text)
        except Exception as e:
            QMessageBox.warning(self, "保存失敗", str(e))

    def refresh(self) -> None:
        # リスト表示は廃止。未確定データをテーブルに再読込。
        try:
            self._load_unconfirmed()
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
        r = requests.post(
            f"{API_URL}/documents/{doc_id}/ok",
            json={
                "date": payload.get("date"),
                "amount": payload.get("amount"),
                "summary": payload.get("summary"),
                "debit_account": payload.get("debit"),
                "credit_account": payload.get("credit"),
                "debit_subaccount": payload.get("debit_sub"),
                "credit_subaccount": payload.get("credit_sub"),
                "invoice_status": payload.get("invoice_status"),
            },
            timeout=20,
        )
        if r.ok:
            QMessageBox.information(self, "保存", "LLM設定を保存しました")
        else:
            QMessageBox.warning(self, "保存失敗", r.text)
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


class CompareDialog(QDialog):
    def __init__(self, company: str, new_doc: dict, old_doc: dict) -> None:
        super().__init__()
        self.setWindowTitle("重複確認")
        layout = QGridLayout()
        layout.addWidget(QLabel("新規"), 0, 0)
        layout.addWidget(QLabel("過去"), 0, 1)

        def _side(v: dict) -> QWidget:
            w = QWidget()
            l = QVBoxLayout()
            l.addWidget(QLabel(f"ID: {v.get('id')}"))
            l.addWidget(QLabel(f"ファイル: {Path(v.get('file_path','')).name}"))
            a = v.get("auto") or {}
            l.addWidget(QLabel(f"日付: {a.get('date')}"))
            l.addWidget(QLabel(f"金額: {a.get('amount')}"))
            l.addWidget(QLabel(f"摘要: {a.get('summary') or ''}"))
            l.addWidget(QLabel(f"借方: {a.get('debit_account') or ''}"))
            l.addWidget(QLabel(f"貸方: {a.get('credit_account') or ''}"))
            # Open buttons
            hb = QHBoxLayout()
            open_btn = QPushButton("PDFを開く")
            open_btn.clicked.connect(lambda: self.open_file(v.get('file_path')))  # type: ignore[arg-type]
            hb.addWidget(open_btn)
            l.addLayout(hb)
            w.setLayout(l)
            return w

        layout.addWidget(_side(new_doc), 1, 0)
        layout.addWidget(_side(old_doc), 1, 1)

        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(self.accept)  # type: ignore[arg-type]
        layout.addWidget(close_btn, 2, 0, 1, 2)
        self.setLayout(layout)

    def open_file(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            import webbrowser
            p = Path(path).resolve()
            webbrowser.open(str(p))
        except Exception:
            pass


class DuplicatesPage(QWidget):
    def __init__(self, company: str) -> None:
        super().__init__()
        self.company = company
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.refresh_btn = QPushButton("更新")
        self.refresh_btn.clicked.connect(self.refresh)  # type: ignore[arg-type]
        layout.addWidget(QLabel("重複の可能性がある候補"))
        layout.addWidget(self.list_widget)
        layout.addWidget(self.refresh_btn)
        self.setLayout(layout)
        self.list_widget.itemActivated.connect(self.on_item)  # type: ignore[arg-type]
        self.refresh()

    def refresh(self) -> None:
        self.list_widget.clear()
        try:
            r = requests.get(f"{API_URL}/duplicates", params={"company_name": self.company}, timeout=10)
            if not r.ok:
                return
            for grp in r.json():
                new = grp.get("new") or {}
                matches = grp.get("matches") or []
                line = f"新規 #{new.get('id')} {Path(new.get('file_path','')).name} → 候補 {len(matches)}件"
                item = QListWidgetItem(line)
                item.setData(256, grp)  # UserRole
                self.list_widget.addItem(item)
        except Exception:
            pass

    def on_item(self, item: QListWidgetItem) -> None:
        data = item.data(256) or {}
        new = data.get("new") or {}
        matches = data.get("matches") or []
        if not matches:
            return
        best = matches[0]
        old = best.get("document") or {}
        dlg = CompareDialog(self.company, new, old)
        dlg.exec()


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
        self.dup_page = DuplicatesPage(company)
        self.settings_page = SettingsPage(company)
        self.stack.addWidget(self.scan_page)
        self.stack.addWidget(self.review_page)
        self.stack.addWidget(self.check_page)
        self.stack.addWidget(self.output_page)
        self.stack.addWidget(self.dup_page)
        self.stack.addWidget(self.settings_page)
        self.setCentralWidget(self.stack)

        # Install account dropdown + autocomplete on Review page table (勘定科目欄)
        def _install_account_inputs() -> None:
            try:
                names, token_map = _load_account_catalog()
                if not names or not hasattr(self, 'review_page'):
                    return
                try:
                    tables = self.review_page.findChildren(QTableWidget)  # type: ignore[attr-defined]
                except Exception:
                    tables = []
                for tbl in tables:
                    try:
                        _install_account_lineedits(tbl, names, token_map)
                    except Exception:
                        continue
            except Exception:
                pass

        _install_account_inputs()

        menubar = self.menuBar()
        # Add top-level actions on menu bar
        # Company selector first
        act_company = menubar.addAction("会社選択")
        act_scan = menubar.addAction("スキャン")
        act_review = menubar.addAction("仕訳確認")
        act_check = menubar.addAction("要チェック")
        act_export = menubar.addAction("出力")
        act_dup = menubar.addAction("重複検知")
        act_settings = menubar.addAction("設定")

        act_company.triggered.connect(self.change_company)  # type: ignore[arg-type]
        act_scan.triggered.connect(self.show_scan)  # type: ignore[arg-type]
        act_review.triggered.connect(self.show_review)  # type: ignore[arg-type]
        act_check.triggered.connect(self.show_check)  # type: ignore[arg-type]
        act_export.triggered.connect(self.show_output)  # type: ignore[arg-type]
        act_dup.triggered.connect(self.show_dup)  # type: ignore[arg-type]
        act_settings.triggered.connect(self.show_settings)  # type: ignore[arg-type]

    def show_scan(self) -> None:
        self.stack.setCurrentIndex(0)

    def show_review(self) -> None:
        self.stack.setCurrentIndex(1)
        # Re-apply delegate-based dropdown in case table was rebuilt
        try:
            names, token_map = _load_account_catalog()
            if names and hasattr(self, 'review_page'):
                for tbl in getattr(self.review_page, 'findChildren', lambda *_: [])(QTableWidget):  # type: ignore[name-defined]
                    try:
                        delegate = AccountCellDelegate(names, token_map, tbl)
                        tbl.setItemDelegateForColumn(1, delegate)
                        tbl.setItemDelegateForColumn(3, delegate)
                    except Exception:
                        pass
        except Exception:
            pass

    def show_check(self) -> None:
        self.stack.setCurrentIndex(2)

    def show_output(self) -> None:
        # Update label to reflect any settings change
        self.output_page.update_info()
        self.stack.setCurrentIndex(3)

    def show_settings(self) -> None:
        idx = self.stack.indexOf(self.settings_page)
        if idx >= 0:
            self.stack.setCurrentIndex(idx)

    def show_dup(self) -> None:
        idx = self.stack.indexOf(self.dup_page)
        if idx >= 0:
            self.stack.setCurrentIndex(idx)

    # admin menu removed; use CompanySelector's 管理ボタンから遷移

    def change_company(self) -> None:
        dlg = CompanySelector()
        res = dlg.exec()
        if res == QDialog.DialogCode.Accepted:
            # If admin requested, open admin window instead of switching
            if getattr(dlg, "admin_requested", False):
                try:
                    admin_win = create_admin_window()
                    admin_win.show()
                except Exception:
                    pass
                return
            if dlg.selected:
                new_company = dlg.selected
                if new_company and new_company != self.company:
                    self.company = new_company
                    # Rebuild pages bound to company
                    new_stack = QStackedWidget()
                    self.scan_page = ScanPage(self.company)
                    self.review_page = ReviewPage(self.company)
                    self.check_page = CheckPage(self.company)
                    self.output_page = OutputPage(self.company)
                    self.dup_page = DuplicatesPage(self.company)
                    self.settings_page = SettingsPage(self.company)
                    new_stack.addWidget(self.scan_page)
                    new_stack.addWidget(self.review_page)
                    new_stack.addWidget(self.check_page)
                    new_stack.addWidget(self.output_page)
                    new_stack.addWidget(self.dup_page)
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
    # No deferred dropdown installer; account inputs are line-edit based and applied when pages build
    sys.exit(app.exec())


if __name__ == "__main__":
    run_ui()





