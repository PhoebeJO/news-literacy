"""選用的 LLM 介面。兩種後端，都沒設定時整個 demo 用規則版，不會呼叫這裡。

設定寫在 news-demo/.env（已列入 .gitignore，不會進版控）：
    LLM_PROVIDER=ollama               ← 本機模型，不需要金鑰（要先裝好 Ollama 並下載模型）
    OLLAMA_MODEL=llama3.1:8b
  或
    GEMINI_API_KEY=你的金鑰            ← 有金鑰時優先用 Gemini（除非 LLM_PROVIDER 指定 ollama）
    GEMINI_MODEL=gemini-2.5-flash     ← 選填；不填會自動挑一個可用的 Flash 模型

自我測試：python -m app.llm
"""
import os
import re

from . import db
from .config import ROOT

_model = None


def _load_env() -> None:
    """讀 .env（不覆蓋已存在的環境變數）。金鑰只放在環境變數，不寫進程式或 Prompt。"""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env()


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def provider() -> str:
    """回傳 'ollama'、'gemini' 或 ''（規則版）。"""
    want = os.getenv("LLM_PROVIDER", "").lower()
    if want in ("none", "rule", "off"):
        return ""
    if want == "ollama":
        return "ollama"
    return "gemini" if os.getenv("GEMINI_API_KEY") else ""


def available() -> bool:
    return bool(provider())


def label() -> str:
    p = provider()
    return {"ollama": f"本機 Ollama {OLLAMA_MODEL}", "gemini": "Gemini"}.get(p, "規則版")


def _log(purpose: str, model: str, tokens_in, tokens_out) -> None:
    with db.connect() as conn:                      # 成本監控：每次呼叫記一筆
        conn.execute(
            "INSERT INTO llm_usage (purpose, model, tokens_in, tokens_out, at) VALUES (?,?,?,?,?)",
            (purpose, model, tokens_in, tokens_out, db.iso(db.now())))


def _ollama(purpose: str, prompt: str, json_mode: bool, max_tokens: int) -> str:
    import httpx
    body = {"model": OLLAMA_MODEL, "stream": False, "keep_alive": "30m",   # 模型留在記憶體，避免每次重新載入
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.2, "num_predict": max_tokens}}
    if json_mode:
        body["format"] = "json"
    r = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    data = r.json()
    _log(purpose, OLLAMA_MODEL, data.get("prompt_eval_count"), data.get("eval_count"))
    return data["message"]["content"]


def _client():
    from google import genai   # 延後匯入：沒裝套件也不影響規則版
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def list_flash_models(client) -> list[str]:
    names = []
    for m in client.models.list():
        name = (m.name or "").removeprefix("models/")
        actions = getattr(m, "supported_actions", None) or []
        if ("flash" in name and "generateContent" in actions
                and not re.search(r"lite|preview|exp|tts|image|live|audio|latest", name)):
            names.append(name)
    return names


def model_name() -> str:
    """GEMINI_MODEL 有設就用；否則從帳號可用的模型裡挑版本最新的一般 Flash。"""
    global _model
    if _model is None:
        _model = os.getenv("GEMINI_MODEL")
        if not _model:
            def version(n: str):
                return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", n)[:1]] or [0.0]
            found = sorted(list_flash_models(_client()), key=version)
            _model = found[-1] if found else "gemini-2.5-flash"
    return _model


def generate(purpose: str, prompt: str, json_mode: bool = False, max_tokens: int = 500) -> str:
    if provider() == "ollama":
        return _ollama(purpose, prompt, json_mode, max_tokens)
    config = {"temperature": 0.2}
    if json_mode:
        config["response_mime_type"] = "application/json"
    model = model_name()
    resp = _client().models.generate_content(model=model, contents=prompt, config=config)
    usage = getattr(resp, "usage_metadata", None)
    _log(purpose, model, getattr(usage, "prompt_token_count", None),
         getattr(usage, "candidates_token_count", None))
    return resp.text or ""


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    import time
    if not available():
        print("沒有設定 LLM：在 news-demo/.env 寫 LLM_PROVIDER=ollama，或 GEMINI_API_KEY=你的金鑰")
        sys.exit(1)
    db.init()
    print("後端：", label())
    if provider() == "gemini":
        print("可用的 Flash 模型：", list_flash_models(_client()))
        print("使用模型：", model_name())
    t = time.time()
    print("測試回覆：", generate("selftest", '只輸出 JSON：{"ok": true}', json_mode=True, max_tokens=20).strip())
    with db.connect() as conn:
        row = conn.execute("SELECT tokens_in, tokens_out FROM llm_usage ORDER BY id DESC LIMIT 1").fetchone()
    print(f"耗時 {time.time() - t:.1f} 秒；token：輸入 {row[0]}、輸出 {row[1]}（已寫入 llm_usage）")
