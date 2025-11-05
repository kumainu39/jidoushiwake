from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


LOGGER = logging.getLogger(__name__)

# OpenAI-compatible Chat Completions endpoint (Colab/Ngrok 等)
API_URL = os.getenv(
    "JIDOU_LLM_CHAT_URL",
    "https://nonbeneficent-oversoftly-piper.ngrok-free.dev/v1/chat/completions",
)


@dataclass
class LLMConfig:
    provider: str = "llama-cpp"
    model_path: Optional[str] = None  # path to GGUF
    device: str = "cpu"  # "cpu" or "gpu"
    n_gpu_layers: int = 0
    n_threads: int = 4
    context_length: int = 4096
    lora_path: Optional[str] = None
    prompt_template: Optional[str] = None
    # CPU 時にリモートHTTP(Colab/Ngrok)を使用するか
    use_colab_remote: bool = False
    # ベースURL（/v1/models, /health 等の liveness 用）
    remote_base_url: str = os.getenv(
        "JIDOU_LLM_REMOTE_BASE",
        "https://nonbeneficent-oversoftly-piper.ngrok-free.dev",
    )


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
    # GPU: available if llama.cpp loads; CPU: remote endpoint must be alive and enabled
    if _CFG.device == "gpu":
        return bool(_load_llama())
    if not _CFG.use_colab_remote:
        return False
    try:
        return _remote_alive()
    except Exception:
        return False


def _remote_alive(timeout: float = 1.5) -> bool:
    """Quick liveness probe for remote endpoint."""
    base = _CFG.remote_base_url.rstrip("/")
    try:
        for ep in ("/health", "/v1/models", "/"):
            url = f"{base}{ep}"
            try:
                r = requests.get(url, timeout=timeout)
                if r.ok:
                    return True
            except Exception:
                continue
        # Many chat servers only implement POST; treat 405 on GET /v1/chat/completions as alive
        try:
            r = requests.get(f"{base}/v1/chat/completions", timeout=timeout)
            if r.status_code == 405:
                return True
        except Exception:
            pass
    except Exception:
        return False
    return False


def _apply_template(base_prompt: str) -> str:
    return _CFG.prompt_template.replace("{{BASE}}", base_prompt) if _CFG.prompt_template else base_prompt


def _chat_complete(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.2,
    stop: Optional[List[str]] = None,
) -> Optional[str]:
    """Call OpenAI-compatible Chat Completions endpoint."""
    try:
        model = os.getenv("JIDOU_LLM_REMOTE_MODEL", "gpt-3.5-turbo")
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        headers = {"Content-Type": "application/json"}
        r = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        if not r.ok:
            return None
        data = r.json()
        chs = data.get("choices") or []
        if chs:
            msg = (chs[0] or {}).get("message") or {}
            txt = msg.get("content")
            if isinstance(txt, str):
                return txt
        # Fallback shape: choices[0].text
        if chs and isinstance(chs[0], dict) and isinstance(chs[0].get("text"), str):
            return chs[0]["text"]
    except Exception:
        return None
    return None


def _http_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.2, stop: Optional[List[str]] = None) -> Optional[str]:
    """Remote HTTP completion helper: try chat first, then a few legacy endpoints."""
    txt = _chat_complete(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
    if isinstance(txt, str) and txt.strip():
        return txt

    base = _CFG.remote_base_url.rstrip("/")
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stop": stop or []}
    headers = {"Content-Type": "application/json"}
    for ep in ("/v1/completions", "/completion", "/generate"):
        url = f"{base}{ep}"
        try:
            r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
            if not r.ok:
                continue
            data = r.json()
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                ch0 = data["choices"][0]
                if isinstance(ch0, dict):
                    txt = ch0.get("text") or (ch0.get("message", {}) or {}).get("content")
                    if isinstance(txt, str):
                        return txt
            if isinstance(data, dict) and isinstance(data.get("response"), str):
                return data["response"]
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                return data["text"]
            if isinstance(data, dict) and isinstance(data.get("results"), list) and data["results"]:
                res0 = data["results"][0]
                if isinstance(res0, dict) and isinstance(res0.get("text"), str):
                    return res0["text"]
        except Exception:
            continue
    return None


def refine_extraction(text_pdf: str, text_paddle: str) -> Optional[Dict[str, Any]]:
    """Use LLM (GPU local or CPU remote) to refine field extraction. Returns dict or None."""
    base_prompt = (
        "以下は領収書・請求書などの日本語OCRテキストです。\n"
        "2種類のOCR結果（PDF埋め込み層、画像OCR）を渡します。\n"
        "取引の基本項目＝日付(YYYY/MM/DD)、金額(整数)、摘要(4文字以上)、借方勘定科目、貸方勘定科目 を推定し、\n"
        "信頼度(0〜1)を含めてJSONで返してください。\n"
        "出力キー: date, amount, summary, debit_account, credit_account, confidence\n"
        "金額は数値のみ。日付はYYYY/MM/DD。未知はnull。\n"
        "[PDF埋め込みテキスト]\n" + (text_pdf or "") + "\n\n"
        "[画像OCR(PaddleOCR)]\n" + (text_paddle or "") + "\n\n"
        "JSONだけを出力してください。"
    )
    prompt = _apply_template(base_prompt)

    if _CFG.device == "gpu":
        llm = _load_llama()
        if not llm:
            return None
        try:
            res = llm.create_completion(prompt=prompt, max_tokens=256, temperature=0.2, stop=["\n\n"])  # type: ignore
            text = res["choices"][0]["text"].strip()
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

    # CPU: remote HTTP
    try:
        if not _CFG.use_colab_remote or not _remote_alive():
            return None
        text = _http_complete(prompt, max_tokens=256, temperature=0.2, stop=["\n\n"]) or ""
        text = text.strip()
        if not text:
            return None
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
        LOGGER.warning("Remote LLM refine failed: %s", e)
        return None


def refine_extraction_with_yomi(text_pdf: str, text_yomitoku: str = "", text_paddle: str = "") -> Optional[Dict[str, Any]]:
    """Refine field extraction using up to three sources: PDF text, YOMITOKU, and image OCR."""
    sections: list[str] = []
    sections.append("[PDF埋め込みテキスト]\n" + (text_pdf or ""))
    if (text_yomitoku or "").strip():
        sections.append("[YOMITOKU(マークダウン結合)]\n" + (text_yomitoku or ""))
    if (text_paddle or "").strip():
        sections.append("[画像OCR(PaddleOCR)]\n" + (text_paddle or ""))

    base_prompt = (
        "以下は領収書・請求書などの日本語OCRテキストです。\n"
        "最大3種類の結果（PDF埋め込み層、YOMITOKU、画像OCR）を渡します。\n"
        "取引の基本項目＝日付(YYYY/MM/DD)、金額(整数)、摘要(4文字以上)、借方勘定科目、貸方勘定科目 を推定し、\n"
        "信頼度(0〜1)を含めてJSONで返してください。\n"
        "出力キー: date, amount, summary, debit_account, credit_account, confidence\n"
        "金額は数値のみ。日付はYYYY/MM/DD。未知はnull。\n"
        + "\n\n".join(sections)
        + "\n\nJSONだけを出力してください。余計な説明は不要です。"
    )
    prompt = _apply_template(base_prompt)

    if _CFG.device == "gpu":
        llm = _load_llama()
        if not llm:
            return None
        try:
            res = llm.create_completion(prompt=prompt, max_tokens=256, temperature=0.2, stop=["\n\n"])  # type: ignore
            text = res["choices"][0]["text"].strip()
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
            LOGGER.warning("LLM refine (with yomi) failed: %s", e)
            return None

    # CPU: remote HTTP
    try:
        if not _CFG.use_colab_remote or not _remote_alive():
            return None
        text = _http_complete(prompt, max_tokens=256, temperature=0.2, stop=["\n\n"]) or ""
        text = text.strip()
        if not text:
            return None
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
        LOGGER.warning("Remote LLM refine (with yomi) failed: %s", e)
        return None


def parse_nl_rule(instruction: str) -> Optional[Dict[str, Any]]:
    """Parse a natural language rule instruction into JSON using LLM when available.

    Returns dict with keys: keyword, debit_account, credit_account, priority, enabled.
    """
    instr = (instruction or "").strip()
    if not instr:
        return None

    if _CFG.device == "gpu":
        llm = _load_llama()
        if llm:
            try:
                base_prompt = (
                    "以下の自然言語の指示から、仕訳の初期設定ルールを抽出し、JSONで返してください。\n"
                    "必須キー: keyword, debit_account, credit_account。任意キー: priority(整数), enabled(真偽)。\n"
                    "キーワードはOCRテキストに含まれる想定の識別語。借方/貸方は勘定科目名。\n"
                    "優先度の既定は0、enabledの既定はtrue。\n"
                    "出力はJSONのみ。余計な説明は不要です。\n"
                    f"指示: {instr}\n"
                )
                prompt = _apply_template(base_prompt)
                res = llm.create_completion(prompt=prompt, max_tokens=200, temperature=0.1, stop=["\n\n"])  # type: ignore
                text = res["choices"][0]["text"].strip()
                data = json.loads(text)
                return {
                    "keyword": (data.get("keyword") or "").strip(),
                    "debit_account": (data.get("debit_account") or "").strip(),
                    "credit_account": (data.get("credit_account") or "").strip(),
                    "priority": int(data.get("priority") or 0),
                    "enabled": bool(data.get("enabled") if data.get("enabled") is not None else True),
                }
            except Exception as e:
                LOGGER.warning("LLM parse_nl_rule failed: %s", e)

    # CPU: remote HTTP
    if _CFG.device == "cpu":
        try:
            if not _CFG.use_colab_remote:
                raise RuntimeError("LLM remote not enabled")
            base_prompt = (
                "以下の自然言語の指示から、仕訳の初期設定ルールを抽出し、JSONで返してください。\n"
                "必須キー: keyword, debit_account, credit_account。任意キー: priority(整数), enabled(真偽)。\n"
                "キーワードはOCRテキストに含まれる想定の識別語。借方/貸方は勘定科目名。\n"
                "優先度の既定は0、enabledの既定はtrue。\n"
                "出力はJSONのみ。余計な説明は不要です。\n"
                f"指示: {instr}\n"
            )
            prompt = _apply_template(base_prompt)
            if not _remote_alive():
                raise RuntimeError("LLM remote not available")
            text = _http_complete(prompt, max_tokens=200, temperature=0.1, stop=["\n\n"]) or ""
            text = text.strip()
            if text:
                data = json.loads(text)
                return {
                    "keyword": (data.get("keyword") or "").strip(),
                    "debit_account": (data.get("debit_account") or "").strip(),
                    "credit_account": (data.get("credit_account") or "").strip(),
                    "priority": int(data.get("priority") or 0),
                    "enabled": bool(data.get("enabled") if data.get("enabled") is not None else True),
                }
        except Exception as e:
            LOGGER.warning("Remote parse_nl_rule failed: %s", e)

    # Fallback heuristic parsing（簡易）
    import re

    def _find_accounts(s: str) -> Tuple[Optional[str], Optional[str]]:
        m = re.search(r"([\w一-龥ぁ-んァ-ヴー・]+)\s*[\/→=＞>\-]+\s*([\w一-龥ぁ-んァ-ヴー・]+)", s)
        if m:
            return m.group(1), m.group(2)
        return None, None

    kw = None
    debit, credit = None, None
    for token in re.split(r"[\s、，。,.]+", instr):
        if not kw and len(token) >= 2:
            kw = token
            break
    d, c = _find_accounts(instr)
    debit = d or debit
    credit = c or credit
    if not kw and not (debit or credit):
        return None
    return {
        "keyword": kw or "",
        "debit_account": debit or "",
        "credit_account": credit or "",
        "priority": 0,
        "enabled": True,
    }

