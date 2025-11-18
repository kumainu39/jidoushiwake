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

# Additional completion endpoints to try after chat-completions.
# You can provide absolute URLs or relative paths (joined to remote_base_url).
_ENV_EXTRA = [
    os.getenv("JIDOU_LLM_GENERATE_URL"),
    os.getenv("JIDOU_LLM_COMPLETION_URL"),
    os.getenv("JIDOU_LLM_V1_COMPLETIONS_URL"),
]
# Comma-separated list takes precedence when set, e.g. "/v1/completions,/completion,/generate"
_CSV = os.getenv("JIDOU_LLM_ENDPOINTS", "").strip()
if _CSV:
    EXTRA_ENDPOINTS: List[str] = [e.strip() for e in _CSV.split(",") if e.strip()]
else:
    # Default: DO NOT probe legacy endpoints unless explicitly configured
    # (Some servers only support chat; probing leads to noisy 404s.)
    EXTRA_ENDPOINTS = [e for e in _ENV_EXTRA if e]


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


_ALIVE_CACHE: dict[str, tuple[float, bool]] = {}


def _remote_alive(timeout: float = 1.5, cache_ttl: float = 90.0) -> bool:
    """Quick liveness probe with short-term caching (reduces noisy GETs).

    - Checks /health and /v1/models; treats 405 on GET /v1/chat/completions as alive.
    - Does not probe '/' to avoid 404 spam.
    """
    import time as _t

    base = _CFG.remote_base_url.rstrip("/")
    key = f"{base}|{','.join(EXTRA_ENDPOINTS)}"
    now = _t.monotonic()
    hit = _ALIVE_CACHE.get(key)
    if hit and (now - hit[0]) < cache_ttl:
        return hit[1]

    ok = False
    try:
        for ep in ("/health", "/v1/models"):
            url = f"{base}{ep}"
            try:
                r = requests.get(url, timeout=timeout)
                if r.ok:
                    ok = True
                    break
            except Exception:
                continue
        if not ok:
            try:
                r = requests.get(f"{base}/v1/chat/completions", timeout=timeout)
                if r.status_code == 405:
                    ok = True
            except Exception:
                pass
        if not ok and EXTRA_ENDPOINTS:
            for ep in EXTRA_ENDPOINTS:
                try:
                    url = ep if ep.startswith("http") else f"{base}{ep}"
                    r = requests.get(url, timeout=timeout)
                    if r.status_code == 405:
                        ok = True
                        break
                except Exception:
                    continue
    except Exception:
        ok = False

    _ALIVE_CACHE[key] = (now, ok)
    return ok


def _apply_template(base_prompt: str) -> str:
    return _CFG.prompt_template.replace("{{BASE}}", base_prompt) if _CFG.prompt_template else base_prompt


def _json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best‑effort JSON extraction from an LLM response.

    - Handles ```json ...``` / ``` ...``` fences
    - Scans multiple JSON objects and picks the best (most non‑null fields)
    - Removes illegal control characters (except TAB/LF/CR)
    - Normalizes "null" (string) → None
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
        # Use shorter connect timeout to avoid UI freeze; allow longer read up to 60s
        read_to = float(os.getenv("JIDOU_LLM_READ_TIMEOUT", "120"))
        r = requests.post(API_URL, json=payload, headers=headers, timeout=(5, read_to))
        if not r.ok:
            return None
        data = r.json()
        chs = data.get("choices") or []
        if chs and isinstance(chs[0], dict):
            # Common shapes
            msg = (chs[0].get("message") or {}) if isinstance(chs[0].get("message"), dict) else {}
            txt = msg.get("content")
            if isinstance(txt, str) and txt.strip():
                return txt
            # Some servers return choices[0].content directly
            if isinstance(chs[0].get("content"), str) and chs[0]["content"].strip():
                return chs[0]["content"]
            # Or choices[0].text
            if isinstance(chs[0].get("text"), str) and chs[0]["text"].strip():
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
    for ep in EXTRA_ENDPOINTS:
        url = ep if ep.startswith("http") else f"{base}{ep}"
        try:
            read_to = float(os.getenv("JIDOU_LLM_READ_TIMEOUT", "120"))
            r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=(5, read_to))
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
        except Exception as e:
            LOGGER.debug("fallback completion failed on %s: %s", url, e)
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
            data = _json_from_text(text)
            if not isinstance(data, dict):
                LOGGER.warning("LLM refine returned non‑JSON (GPU): %s", text[:200])
                return None
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
        text = (_http_complete(prompt, max_tokens=256, temperature=0.2, stop=["\n\n"]) or "").strip()
        if not text:
            return None
        data = _json_from_text(text)
        if not isinstance(data, dict):
            LOGGER.warning("Remote LLM refine returned non‑JSON: %s", text[:200])
            return None
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
            data = _json_from_text(text)
            if not isinstance(data, dict):
                LOGGER.warning("LLM refine (with yomi) returned non‑JSON (GPU): %s", text[:200])
                return None
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
        text = (_http_complete(prompt, max_tokens=256, temperature=0.2, stop=["\n\n"]) or "").strip()
        if not text:
            return None
        data = _json_from_text(text)
        if not isinstance(data, dict):
            LOGGER.warning("Remote LLM refine (with yomi) returned non‑JSON: %s", text[:200])
            return None
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
