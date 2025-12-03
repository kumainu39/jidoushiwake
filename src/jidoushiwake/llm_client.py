from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = "llama-cpp"
    model_path: Optional[str] = "F:\\models\\Llama-3-ELYZA-JP-8B-q4_k_m.gguf"  # path to GGUF
    device: str = "gpu"  # GPU is required; CPU fallback is not supported
    n_gpu_layers: int = -1  # -1 = auto/full offload
    n_threads: int = 4  # kept for llama.cpp compatibility
    context_length: int = 4096
    lora_path: Optional[str] = None
    prompt_template: Optional[str] = None


_LLM: Any | None = None
_CFG = LLMConfig()


def set_config(cfg: LLMConfig) -> None:
    global _CFG, _LLM
    _CFG = cfg
    _LLM = None  # force reload with new settings


class _TempConfig:
    def __init__(self, new_cfg: LLMConfig) -> None:
        self.new_cfg = new_cfg
        self.old_cfg = _CFG

    def __enter__(self):
        set_config(self.new_cfg)
        return self

    def __exit__(self, exc_type, exc, tb):
        set_config(self.old_cfg)


def temporary_config(cfg: LLMConfig):
    return _TempConfig(cfg)


def _gpu_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_llama() -> Any | None:
    global _LLM
    if _LLM is not None:
        return _LLM
    if _CFG.provider != "llama-cpp" or not _CFG.model_path:
        return None
    if not _gpu_available():
        LOGGER.error("GPU device is required but not available.")
        return None
    try:
        from llama_cpp import Llama  # type: ignore

        model_path = Path(_CFG.model_path)
        if not model_path.exists():
            LOGGER.error("LLM model path does not exist: %s", model_path)
            return None

        n_gpu_layers = _CFG.n_gpu_layers if _CFG.n_gpu_layers not in (None, 0) else -1
        kwargs = dict(
            model_path=str(model_path),
            n_ctx=_CFG.context_length,
            n_threads=_CFG.n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        if _CFG.lora_path:
            kwargs["lora_path"] = _CFG.lora_path
        _LLM = Llama(**kwargs)
        LOGGER.info("Loaded llama.cpp model on GPU. n_gpu_layers=%d", n_gpu_layers)
        return _LLM
    except Exception as e:
        LOGGER.error("Failed to load llama model on GPU: %s", e)
        return None


def available() -> bool:
    # Report whether the local model is ready (GPU preferred, CPU fallback allowed).
    return bool(_load_llama())


def _apply_template(base_prompt: str) -> str:
    return _CFG.prompt_template.replace("{{BASE}}", base_prompt) if _CFG.prompt_template else base_prompt


def _json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from an LLM response.

    - Handles ```json ...``` / ``` ...``` fences
    - Scans multiple JSON objects and picks the best (most non-null fields)
    - Removes illegal control characters (except TAB/LF/CR)
    - Normalizes "null" (string) to None
    """
    import re

    def _clean(s: str) -> str:
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)

    def _score_and_norm(d: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        keys = ("date", "amount", "summary", "debit_account", "credit_account")
        norm = {}
        filled = 0
        for k in keys:
            v = d.get(k)
            if isinstance(v, str) and v.strip().lower() == "null":
                v = None
            norm[k] = v
            if v not in (None, ""):
                filled += 1
        # keep other keys if present
        for k, v in d.items():
            if k not in norm:
                norm[k] = v
        return filled, norm

    if not isinstance(text, str):
        return None
    src = text.strip()
    if not src:
        return None

    # Prefer fenced block
    m = re.search(r"```\s*json\s*(.*?)\s*```", src, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```\s*(.*?)\s*```", src, flags=re.DOTALL)
    if m:
        cand = _clean(m.group(1).strip())
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                return _score_and_norm(d)[1]
        except Exception:
            pass

    # Otherwise, scan all {...} blocks and pick the best
    blocks: List[str] = []
    buf = []
    depth = 0
    for ch in src:
        if ch == '{':
            depth += 1
        if depth > 0:
            buf.append(ch)
        if ch == '}':
            depth -= 1
            if depth == 0 and buf:
                blocks.append(_clean(''.join(buf)))
                buf = []
    best: Tuple[int, Dict[str, Any]] | None = None
    for b in blocks:
        try:
            d = json.loads(b)
            if not isinstance(d, dict):
                continue
            sc, norm = _score_and_norm(d)
            if (best is None) or (sc > best[0]):
                best = (sc, norm)
        except Exception:
            continue
    return best[1] if best else None


def _complete(prompt: str, max_tokens: int, temperature: float, stop: Optional[List[str]]) -> Optional[str]:
    llm = _load_llama()
    if not llm:
        LOGGER.error("GPU LLM not available for completion.")
        return None
    try:
        res = llm.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)  # type: ignore
        txt = res["choices"][0]["text"]
        return txt if isinstance(txt, str) else None
    except Exception as e:
        LOGGER.warning("LLM completion failed: %s", e)
        return None


def refine_extraction(text_pdf: str, text_paddle: str) -> Optional[Dict[str, Any]]:
    """Use the local GPU LLM to refine field extraction. Returns dict or None."""
    base_prompt = (
        "以下�E領収書・請求書などの日本語OCRチE��ストです、En"
        "2種類�EOCR結果�E�EDF埋め込み層、画像OCR�E�を渡します、En"
        "取引�E基本頁E���E�日仁EYYYY/MM/DD)、E��顁E整数)、摘要E4斁E��以丁E、借方勘定科目、貸方勘定科目 を推定し、\n"
        "信頼度(0、E)を含めてJSONで返してください、En"
        "出力キー: date, amount, summary, debit_account, credit_account, confidence\n"
        "金額�E数値のみ。日付�EYYYY/MM/DD。未知はnull、En"
        "[PDF埋め込みチE��スチE\n" + (text_pdf or "") + "\n\n"
        "[画像OCR(PaddleOCR)]\n" + (text_paddle or "") + "\n\n"
        "JSONだけを出力してください、E"
    )
    prompt = _apply_template(base_prompt)

    text = _complete(prompt, max_tokens=256, temperature=0.2, stop=["\n\n"])
    if not text:
        return None
    data = _json_from_text(text.strip())
    if not isinstance(data, dict):
        LOGGER.warning("LLM refine returned non-JSON: %s", text[:200])
        return None
    return {
        "date": data.get("date"),
        "amount": data.get("amount"),
        "summary": data.get("summary"),
        "debit_account": data.get("debit_account"),
        "credit_account": data.get("credit_account"),
        "confidence": data.get("confidence"),
    }


def refine_extraction_with_yomi(text_pdf: str, text_yomitoku: str = "", text_paddle: str = "") -> Optional[Dict[str, Any]]:
    """Refine field extraction using up to three sources: PDF text, YOMITOKU, and image OCR."""
    sections: list[str] = []
    sections.append("[PDF埋め込みチE��スチE\n" + (text_pdf or ""))
    if (text_yomitoku or "").strip():
        sections.append("[YOMITOKU(マ�Eクダウン結合)]\n" + (text_yomitoku or ""))
    if (text_paddle or "").strip():
        sections.append("[画像OCR(PaddleOCR)]\n" + (text_paddle or ""))

    base_prompt = (
        "以下�E領収書・請求書などの日本語OCRチE��ストです、En"
        "最大3種類�E結果�E�EDF埋め込み層、YOMITOKU、画像OCR�E�を渡します、En"
        "取引�E基本頁E���E�日仁EYYYY/MM/DD)、E��顁E整数)、摘要E4斁E��以丁E、借方勘定科目、貸方勘定科目 を推定し、\n"
        "信頼度(0、E)を含めてJSONで返してください、En"
        "出力キー: date, amount, summary, debit_account, credit_account, confidence\n"
        "金額�E数値のみ。日付�EYYYY/MM/DD。未知はnull、En"
        + "\n\n".join(sections)
        + "\n\nJSONだけを出力してください。余計な説明�E不要です、E"
    )
    prompt = _apply_template(base_prompt)

    text = _complete(prompt, max_tokens=256, temperature=0.2, stop=["\n\n"])
    if not text:
        return None
    data = _json_from_text(text.strip())
    if not isinstance(data, dict):
        LOGGER.warning("LLM refine (with yomi) returned non-JSON: %s", text[:200])
        return None
    return {
        "date": data.get("date"),
        "amount": data.get("amount"),
        "summary": data.get("summary"),
        "debit_account": data.get("debit_account"),
        "credit_account": data.get("credit_account"),
        "confidence": data.get("confidence"),
    }


def parse_nl_rule(instruction: str) -> Optional[Dict[str, Any]]:
    """Parse a natural language rule instruction into JSON using the GPU LLM."""

    instr = (instruction or "").strip()
    if not instr:
        return None

    text = None
    try:
        base_prompt = (
            "以下�E自然言語�E持E��から、仕訳の初期設定ルールを抽出し、JSONで返してください、En"
            "忁E��キー: keyword, debit_account, credit_account。任意キー: priority(整数), enabled(真偽)、En"
            "キーワード�EOCRチE��ストに含まれる想定�E識別語。借方/貸方は勘定科目名、En"
            "優先度の既定�E0、enabledの既定�Etrue、En"
            "出力�EJSONのみ。余計な説明�E不要です、En"
            f"持E��: {instr}\n"
        )
        prompt = _apply_template(base_prompt)
        text = _complete(prompt, max_tokens=200, temperature=0.1, stop=["\n\n"])
    except Exception as e:
        LOGGER.warning("LLM parse_nl_rule failed: %s", e)
        return None

    if not text:
        return None
    try:
        data = json.loads(text.strip())
        return {
            "keyword": (data.get("keyword") or "").strip(),
            "debit_account": (data.get("debit_account") or "").strip(),
            "credit_account": (data.get("credit_account") or "").strip(),
            "priority": int(data.get("priority") or 0),
            "enabled": bool(data.get("enabled") if data.get("enabled") is not None else True),
        }
    except Exception as e:
        LOGGER.warning("LLM parse_nl_rule JSON decode failed: %s", e)
        return None
