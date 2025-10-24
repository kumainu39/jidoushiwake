from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = "llama-cpp"
    model_path: Optional[str] = None  # path to GGUF (elyza/Llama-3-ELYZA-JP-8B-GGUF)
    device: str = "cpu"  # "cpu" or "gpu"
    n_gpu_layers: int = 0  # >0 enables GPU offload
    n_threads: int = 4
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
    try:
        from llama_cpp import Llama  # type: ignore

        model_path = Path(_CFG.model_path)
        if not model_path.exists():
            LOGGER.warning("LLM model path does not exist: %s", model_path)
            return None

        n_gpu_layers = _CFG.n_gpu_layers if (_CFG.device == "gpu" and _gpu_available()) else 0
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
        LOGGER.info("Loaded llama.cpp model. device=%s n_gpu_layers=%d", _CFG.device, n_gpu_layers)
        return _LLM
    except Exception as e:
        LOGGER.error("Failed to load llama model: %s", e)
        return None


def available() -> bool:
    # Heavy usage only when device is gpu and GPU available
    if _CFG.device != "gpu":
        return False
    return bool(_load_llama())


def refine_extraction(text_pdf: str, text_paddle: str) -> Optional[Dict[str, Any]]:
    """Use LLM to refine field extraction. Returns dict or None on failure.

    Only runs when device is GPU and model is loaded.
    """
    llm = _load_llama()
    if not llm or _CFG.device != "gpu":
        return None

    base_prompt = (
        "以下は領収書・請求書などの日本語OCRテキストです。\n"
        "2種類のOCR結果（PDF埋め込み層、画像OCR）を渡します。\n"
        "取引の日付(YYYY/MM/DD)、金額(整数, 円)、摘要(64文字以内)、借方勘定科目、貸方勘定科目を推定し、\n"
        "信頼度(0から1)を含めてJSONで返してください。\n"
        "出力キー: date, amount, summary, debit_account, credit_account.\n"
        "金額は数値のみ。日付はYYYY/MM/DD。未知はnull。confidenceは0.0〜1.0。\n\n"
        "[PDF埋め込みテキスト]\n" + (text_pdf or "") + "\n\n"
        "[画像OCR(PaddleOCR)]\n" + (text_paddle or "") + "\n\n"
        "JSONだけを出力してください。"
    )
    if _CFG.prompt_template:
        prompt = _CFG.prompt_template.replace("{{BASE}}", base_prompt)
    else:
        prompt = base_prompt

    try:
        res = llm.create_completion(
            prompt=prompt,
            max_tokens=256,
            temperature=0.2,
            stop=["\n\n"]
        )
        text = res["choices"][0]["text"].strip()
        import json

        data = json.loads(text)
        return {
            "date": data.get("date"),
            "amount": data.get("amount"),
            "summary": data.get("summary"),
            "debit_account": data.get("debit_account"),
            "credit_account": data.get("credit_account"),
            "confidence": data.get("confidence"),
        }
    except Exception as e:
        LOGGER.warning("LLM refine failed: %s", e)
        return None
