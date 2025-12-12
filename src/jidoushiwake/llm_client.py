from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)

# Fixed model path (GPU llama.cpp only)
DEFAULT_MODEL_PATH = Path(r"F:\models\Llama-3-ELYZA-JP-8B-q4_k_m.gguf")

# RAG-like static knowledge for journal inference (実際の領収書を想定したキーワードと勘定)
KNOWLEDGE_JOURNAL = """
[仕訳ルール参照(RAG)] 伝票・領収書で実際に現れる店名や表記を想定
- 給油/ガソリン/燃料: ENEOS, 出光, コスモ, エネオスSS, Shell, シェル, キグナス, 昭和シェル, モービル → 借方=車両費(燃料費) or 燃料費 / 貸方=決済手段
- 高速/ETC/駐車場: ETC利用照会, 駐車料金, Times, NPC, 三井のリパーク → 借方=旅費交通費(高速/駐車場) / 貸方=決済手段
- 交通系IC/乗車券/タクシー: スイカ/ Suica, PASMO, ICOCA, タクシー領収書(○○交通, ○○ハイヤー), JR券売機, 新幹線, バス回数券 → 借方=旅費交通費 / 貸方=決済手段
- 交通系IC/乗車券: スイカ/ Suica, PASMO, ICOCA, JR券売機, 新幹線, バス回数券 → 借方=旅費交通費 / 貸方=決済手段
- 外食/会食/飲食店: 居酒屋, レストラン, ○○食堂, ○○寿司, ○○ラーメン, 喫茶, スターバックス, ドトール, コメダ, サイゼリヤ → 借方=交際費（社外対応） or 会議費（社内少額） / 貸方=決済手段
- 小売/日用品/文具: セブン‐イレブン, ファミリーマート, ローソン, ミニストップ, 東急ハンズ, ロフト, コクヨ, カインズ, コーナン, ビバホーム, Daiso, Seria → 少額消耗なら借方=消耗品費 / 貸方=決済手段
- 事務用品/印刷: プリンタ, インク, トナー, コピー用紙, 印刷代, 名刺印刷 → 借方=消耗品費 or 事務用品費 / 貸方=決済手段
- 通信/携帯/回線: NTTドコモ, au, ソフトバンク, 楽天モバイル, フレッツ, ひかり, Wi-Fi, プロバイダ → 借方=通信費 / 貸方=決済手段
- クラウド/ITサービス/サブスク: Google, Microsoft, Adobe, AWS, Azure, GCP, Slack, Zoom, Notion, Dropbox → 借方=通信費 or ソフトウェア / 貸方=決済手段
- 配送/郵便/宅配: ヤマト運輸, 佐川急便, 日本郵便, ゆうパック, レターパック, クリックポスト, DHL, FedEx → 借方=通信運搬費 / 貸方=決済手段
- 広告/マーケ出稿: Meta広告, Facebook広告, Google広告, X(旧Twitter)広告, LINE広告, リスティング, バナー → 借方=広告宣伝費 / 貸方=決済手段
- オフィス賃料/スペース: 賃料, 家賃, 共益費, WeWork, レンタルオフィス, 会議室レンタル → 借方=地代家賃 / 貸方=決済手段
- 光熱: 電気, ガス, 水道, 検針票/請求書, ○○電力, ○○ガス, ○○水道局 → 借方=水道光熱費 / 貸方=決済手段
- 修繕/保守/クリーニング: 修理, 保守, 点検, クリーニング, 設備メンテ → 借方=修繕費 / 貸方=決済手段
- 教育/研修/書籍: セミナー, 研修, 受講料, ○○カレッジ, 書籍, 技術書, 勉強会 → 借方=研修費 or 図書研修費 / 貸方=決済手段
- 宿泊/出張: ホテル, ビジネスホテル, 宿泊税, Airbnb, 旅館 → 借方=旅費交通費(宿泊費) / 貸方=決済手段
- 決済手段の基本: 現金→貸方=現金、銀行引落/振込→貸方=普通預金、クレカ→貸方=未払金(カード名等)、振込受取→借方=普通預金
"""


@dataclass
class LLMConfig:
    provider: str = "llama-cpp"
    model_path: Optional[str] = str(DEFAULT_MODEL_PATH)  # path to GGUF
    device: str = "gpu"  # GPU is required; CPU fallback is not supported
    n_gpu_layers: int = -1  # -1 = auto/full offload
    n_threads: int = 4  # kept for llama.cpp compatibility
    context_length: int = 8192
    lora_path: Optional[str] = None
    prompt_template: Optional[str] = None


_LLM: Any | None = None
_CFG = LLMConfig()


def set_config(cfg: LLMConfig) -> None:
    global _CFG, _LLM
    # Force GPU llama.cpp with fixed model path; ignore CPU/other providers during tests
    cfg.provider = "llama-cpp"
    cfg.device = "gpu"
    if not cfg.model_path:
        cfg.model_path = str(DEFAULT_MODEL_PATH)
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
    # Always use local llama.cpp GGUF; no CPU fallback
    if _CFG.provider != "llama-cpp":
        LOGGER.error("Only llama-cpp GPU provider is supported in this build.")
    model_path = Path(_CFG.model_path or DEFAULT_MODEL_PATH)
    if not model_path:
        LOGGER.error("LLM model path is not set.")
        return None
    if not _gpu_available():
        LOGGER.error("GPU device is required but not available.")
        return None
    try:
        from llama_cpp import Llama  # type: ignore
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
                filled += 2 if k == "amount" else 1  # prefer candidates that include amount
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


def _enforce_taxi_rule(src_texts: List[str], data: Dict[str, Any]) -> None:
    """Override debit to taxi fare when taxi cues are present."""
    text_blob = " ".join(t for t in src_texts if t)
    flat = text_blob.replace(" ", "").replace("\n", "").replace("\t", "")
    import re as _re
    taxi_keywords = [
        "タクシー",
        "ﾀｸｼｰ",
        "タクシ",
        "ﾀｸｼ",
        "メータ",
        "迎車",
        "配車",
        "運賃",
        "ハイヤー",
        "交通株式会社",
        "タクシー株式会社",
    ]
    has_taxi = any(k in text_blob for k in taxi_keywords)
    has_taxi = has_taxi or "交通株" in flat or bool(_re.search(r"交通\\s*株", text_blob)) or bool(_re.search(r"交通.*(株|株式会社)", text_blob))
    if has_taxi:
        data["debit_account"] = "旅費交通費"
        # If summary is missing or短すぎ, append taxi hint
        summary = (data.get("summary") or "").strip()
        if "タクシ" not in summary:
            data["summary"] = (summary + " タクシー利用").strip()


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
    # Ignore embedded PDF text and PaddleOCR; rely on upstream OCR only
    src_texts = [text_pdf or "", text_paddle or ""]
    text_pdf = ""
    text_paddle = ""
    base_prompt = (
        "以下は領収書・請求書などの日本語OCRテキストです。\n"
        "【最優先】タクシー/メータ運賃/迎車/配車/タクシー会社名（○○交通/○○タクシー/○○ハイヤー/○○交通株式会社など）が1つでもあれば、借方=旅費交通費 とし、摘要にもタクシー利用を含める。\n"
        "取引の基本項目（日付YYYY/MM/DD、金額=整数）、摘要（発行元の会社名/店名を含め4〜30文字程度）、借方勘定科目、貸方勘定科目を推定し、\n"
        "消費税額と税率（8/10など）が読み取れれば tax_amount, tax_rate も返してください。\n"
        "仕訳の基本ルール: 決済手段が不明なときは常に credit=現金 とする。クレカ払が読み取れたら credit=未払金(カード名等) とし、それ以外で預金を使うことはない。商品/サービスの内容に応じて debit を選ぶ（例: ガソリン/交通→車両費/旅費交通費、飲食→交際費、文具/用品→消耗品費）。タクシー/メータ運賃/○○交通株式会社/○○ハイヤー/○○タクシー/迎車/配車などがあれば必ず借方=旅費交通費 とする。\n"
        "インボイス判定も返してください。知識: インボイス登録番号は先頭\"T\"+13桁(例: T1234567890123)。ハイフン区切り(T1-2345-6789-0123)も同じ番号として扱い、番号があれば invoice_status=\"適格\"。領収書/請求書等の記載はあるが番号が無ければ \"非適格\"。非課税/不課税/対象外などの語があれば \"非課税\"。判断不能は null。\n"
        + KNOWLEDGE_JOURNAL + "\n"
        "信頼度(0-1)を含めてJSONで返してください。\n"
        "出力キー: date, amount, tax_amount, tax_rate, summary, issuer, debit_account, credit_account, confidence, invoice_status\n"
        "金額は数値のみ。日付はYYYY/MM/DD。税率は整数(8/10など)。未知はnull。\n"
        "[OCRテキスト]\n" + (text_pdf or "") + "\n\n"
        "JSONだけを出力してください。余計な説明は不要です。"
    )
    prompt = _apply_template(base_prompt)

    text = _complete(prompt, max_tokens=256, temperature=0.2, stop=["\n\n"])
    if not text:
        return None
    data = _json_from_text(text.strip())
    if not isinstance(data, dict):
        LOGGER.warning("LLM refine returned non-JSON: %s", text[:200])
        return None
    _enforce_taxi_rule(src_texts + [str(data.get("summary") or ""), str(data.get("issuer") or "")], data)
    return {
        "date": data.get("date"),
        "amount": data.get("amount"),
        "tax_amount": data.get("tax_amount"),
        "tax_rate": data.get("tax_rate"),
        "summary": data.get("summary"),
        "issuer": data.get("issuer"),
        "debit_account": data.get("debit_account"),
        "credit_account": data.get("credit_account"),
        "confidence": data.get("confidence"),
        "invoice_status": data.get("invoice_status"),
    }


def refine_extraction_with_yomi(text_pdf: str, text_yomitoku: str = "", text_paddle: str = "") -> Optional[Dict[str, Any]]:
    """Refine field extraction using up to three sources: PDF text, YOMITOKU, and image OCR."""
    # Ignore embedded PDF text and PaddleOCR; rely on upstream OCR only
    src_texts = [text_pdf or "", text_yomitoku or "", text_paddle or ""]
    text_pdf = ""
    text_paddle = ""
    sections: list[str] = []
    if (text_yomitoku or "").strip():
        sections.append("[YOMITOKU(マークダウン結合)]\n" + (text_yomitoku or ""))

    base_prompt = (
        "以下は領収書・請求書などの日本語OCRテキストです。\n"
        "【最優先】タクシー/メータ運賃/迎車/配車/タクシー会社名（○○交通/○○タクシー/○○ハイヤー/○○交通株式会社など）が1つでもあれば、借方=旅費交通費 とし、摘要にもタクシー利用を含める。\n"
        "取引の基本項目（日付YYYY/MM/DD、金額=整数）、摘要（発行元の会社名/店名を含め4〜30文字程度）、借方勘定科目、貸方勘定科目を推定し、\n"
        "消費税額と税率（8/10など）が読み取れれば tax_amount, tax_rate も返してください。\n"
        "仕訳の基本ルール: 決済手段が不明なときは常に credit=現金 とする。クレカ払が読み取れたら credit=未払金(カード名等) とし、それ以外で預金を使うことはない。商品/サービスの内容に応じて debit を選ぶ（例: ガソリン/交通→車両費/旅費交通費、飲食→交際費、文具/用品→消耗品費）。タクシー/メータ運賃/○○交通株式会社/○○ハイヤー/○○タクシー/迎車/配車などがあれば必ず借方=旅費交通費 とする。\n"
        "インボイス判定も返してください。知識: インボイス登録番号は先頭\"T\"+13桁(例: T1234567890123)。ハイフン区切り(T1-2345-6789-0123)も同じ番号として扱い、番号があれば invoice_status=\"適格\"。領収書/請求書等の記載はあるが番号が無ければ \"非適格\"。非課税/不課税/対象外などの語があれば \"非課税\"。判断不能は null。\n"
        + KNOWLEDGE_JOURNAL + "\n"
        "信頼度(0-1)を含めてJSONで返してください。\n"
        "出力キー: date, amount, tax_amount, tax_rate, summary, issuer, debit_account, credit_account, confidence, invoice_status\n"
        "金額は数値のみ。日付はYYYY/MM/DD。税率は整数(8/10など)。未知はnull。\n"
        + "\n\n".join(sections)
        + "\n\nJSONだけを出力してください。余計な説明は不要です。"
    )
    prompt = _apply_template(base_prompt)

    text = _complete(prompt, max_tokens=256, temperature=0.2, stop=["\n\n"])
    if not text:
        return None
    data = _json_from_text(text.strip())
    if not isinstance(data, dict):
        LOGGER.warning("LLM refine (with yomi) returned non-JSON: %s", text[:200])
        return None
    _enforce_taxi_rule(src_texts + [str(data.get("summary") or ""), str(data.get("issuer") or "")], data)
    return {
        "date": data.get("date"),
        "amount": data.get("amount"),
        "tax_amount": data.get("tax_amount"),
        "tax_rate": data.get("tax_rate"),
        "summary": data.get("summary"),
        "issuer": data.get("issuer"),
        "debit_account": data.get("debit_account"),
        "credit_account": data.get("credit_account"),
        "confidence": data.get("confidence"),
        "invoice_status": data.get("invoice_status"),
    }


def parse_nl_rule(instruction: str) -> Optional[Dict[str, Any]]:
    """Parse a natural language rule instruction into JSON using the GPU LLM."""

    instr = (instruction or "").strip()
    if not instr:
        return None

    text = None
    try:
        base_prompt = (
            "以下の自然言語の指示から、仕訳の初期設定ルールを抽出し、JSONで返してください。\n"
            "必須キー: keyword, debit_account, credit_account。任意キー: priority(整数), enabled(真偽)\n"
            "キーワードはOCRテキストに含まれる想定の識別語。借方/貸方は勘定科目名。\n"
            "優先度の既定は0、enabledの既定はtrue。\n"
            "出力はJSONのみ。余計な説明は不要です。\n"
            f"指示: {instr}\n"
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
