from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

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
    # When device=="cpu", optionally call remote HTTP endpoint (e.g., Colab/Ngrok)
    use_colab_remote: bool = False
    remote_base_url: str = os.getenv("JIDOU_LLM_REMOTE_BASE", "http://localhost:8005")


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
    # GPU: available if llama.cpp loads; CPU: assume remote endpoint is used
    if _CFG.device == "gpu":
        return bool(_load_llama())
    # CPU: only consider remote endpoint when explicitly enabled
    if not _CFG.use_colab_remote:
        return False
    try:
        return _remote_alive()
    except Exception:
        return False


def _remote_alive(timeout: float = 1.5) -> bool:
    """Quickly probe remote endpoint to avoid long timeouts when offline.

    Compatible with:
      - OpenAI-like servers exposing GET /v1/models or GET /health
      - Minimal FastAPI servers that only implement POST /generate
        (we treat a 405 on GET /generate as a positive liveness signal)
    """
    base = _CFG.remote_base_url.rstrip("/")
    try:
        # Common health-style endpoints
        for ep in ("/health", "/v1/models", "/"):
            url = f"{base}{ep}"
            try:
                r = requests.get(url, timeout=timeout)
                if r.ok:
                    return True
            except Exception:
                continue
        # If only POST /generate exists, a GET often returns 405; treat that as alive
        try:
            r = requests.get(f"{base}/generate", timeout=timeout)
            if r.status_code == 405:
                return True
        except Exception:
            pass
    except Exception:
        return False
    return False


def _apply_template(base_prompt: str) -> str:
    return _CFG.prompt_template.replace("{{BASE}}", base_prompt) if _CFG.prompt_template else base_prompt


def _http_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.2, stop: Optional[List[str]] = None) -> Optional[str]:
    """Call remote HTTP LLM for completion.

    Tries a few common endpoints; expects one of the following response shapes:
      - OpenAI-like: { choices: [{ text: "..." }] }
      - Simple: { text: "..." }
      - TGI/text-gen: { results: [{ text: "..." }] }
    """
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
            # Simple Colab server shape: { "response": "..." }
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
        "以下の領収書・請求書などの日本語OCRテキストです。\n"
        "2種類のOCR結果（PDF埋め込み層、画像OCR）を渡します。\n"
        "取引の基本項目（日付YYYY/MM/DD、金額整数、摘要64文字以内、借方勘定科目、貸方勘定科目）を推定し、\n"
        "信頼度(0から1)を含めてJSONで返してください。\n"
        "出力キー: date, amount, summary, debit_account, credit_account.\n"
        "金額は数値のみ。日付はYYYY/MM/DD。未知はnull。confidenceは0.0-1.0。\n\n"
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

    # CPU: remote HTTP (Colab/tunnel) only when enabled. Probe first to avoid long hangs when offline.
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
    """Refine field extraction using up to three sources: PDF text, YOMITOKU, and image OCR.

    Prefer YOMITOKU when available by presenting it explicitly to the LLM.
    Returns dict or None.
    """
    sections: list[str] = []
    sections.append("[PDF埋め込みテキスト]\n" + (text_pdf or ""))
    if (text_yomitoku or "").strip():
        sections.append("[YOMITOKU(マークダウン結合)]\n" + (text_yomitoku or ""))
    if (text_paddle or "").strip():
        sections.append("[画像OCR(PaddleOCR)]\n" + (text_paddle or ""))

    base_prompt = (
        "以下は領収書・請求書などの日本語OCRテキストです。\n"
        "最大3種類の結果（PDF埋め込み層、YOMITOKU、画像OCR）を渡します。\n"
        "取引の基本項目（＝日付YYYY/MM/DD、金額整数、摘要64字以内、借方勘定科目、貸方勘定科目）を推定し、\n"
        "信頼度(0から1)を含めてJSONで返してください。\n"
        "出力キー: date, amount, summary, debit_account, credit_account。\n"
        "金額は数値のみ。日付はYYYY/MM/DD。未知はnull。confidenceは0.0-1.0。\n\n"
        + "\n\n".join(sections)
        + "\n\nJSONだけを出力してください。余計な説明は不要です。\n"
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

    # CPU: remote HTTP (Colab/tunnel) only when enabled. Probe first to avoid long hangs when offline.
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
    """Parse a natural language rule instruction into JSON using LLM when available,
    otherwise a heuristic fallback.
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
                    "必須キー: keyword, debit_account, credit_account. 任意キー: priority(整数), enabled(真偽).\n"
                    "キーワードはOCRテキストに含まれる想定の識別語。借方/貸方は勘定科目名。\n"
                    "優先度の既定は0、enabledの既定はtrue。\n"
                    "出力はJSONのみ。余計な説明を付けない。\n\n"
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
                "必須キー: keyword, debit_account, credit_account. 任意キー: priority(整数), enabled(真偽).\n"
                "キーワードはOCRテキストに含まれる想定の識別語。借方/貸方は勘定科目名。\n"
                "優先度の既定は0、enabledの既定はtrue。\n"
                "出力はJSONのみ。余計な説明を付けない。\n\n"
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

    # Fallback heuristic parsing
    import re

    def _find_accounts(s: str) -> Tuple[Optional[str], Optional[str]]:
        m = re.search(r"([\w一-龥ぁ-んァ-ヴー・]+)\s*[\/→⇒=>]\s*([\w一-龥ぁ-んァ-ヴー・]+)", s)
        if m:
            return m.group(1), m.group(2)
        md = re.search(r"借方(?:科目)?[:：]?\s*([\w一-龥ぁ-んァ-ヴー・]+)", s)
        mc = re.search(r"貸方(?:科目)?[:：]?\s*([\w一-龥ぁ-んァ-ヴー・]+)", s)
        return (md.group(1) if md else None, mc.group(1) if mc else None)

    def _find_keyword(s: str) -> Optional[str]:
        q = re.search(r"[『「\"]([^『「\"]+)[』」\"]", s)
        if q:
            return q.group(1).strip()
        h = re.search(r"([\w一-龥ぁ-んァ-ヴー・]+)は", s)
        if h:
            return h.group(1).strip()
        t = re.search(r"([\w一-龥ぁ-んァ-ヴー・]{3,})", s)
        return t.group(1).strip() if t else None

    debit, credit = _find_accounts(instr)
    keyword = _find_keyword(instr)
    prio = 0
    en = True
    mprio = re.search(r"優先度[:：]?\s*(-?\d+)", instr)
    if mprio:
        try:
            prio = int(mprio.group(1))
        except Exception:
            pass
    if re.search(r"無効|disable|off", instr, re.IGNORECASE):
        en = False

    if not keyword or not debit or not credit:
        return None
    return {"keyword": keyword, "debit_account": debit, "credit_account": credit, "priority": prio, "enabled": en}
