import json
import time
import ssl
import urllib.request
import urllib.parse
import urllib.error
from .config import PRICING, SONNET_ID, GEMINI_FLASH_ID, GCP_PROJECT, GCP_LOCATION

_vertex_token = {"tok": None, "exp": 0.0}

def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def _vertex_access_token():
    if _vertex_token["tok"] and _vertex_token["exp"] - 60 > time.time():
        return _vertex_token["tok"]
    import google.auth
    import google.auth.transport.requests
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    _vertex_token["tok"] = creds.token
    _vertex_token["exp"] = creds.expiry.timestamp() if creds.expiry else time.time() + 3000
    return creds.token

_CLAUDE_THINKING_BUDGET = {"low": 2048, "medium": 4096, "high": 8192}

def claude_api_call(model_id, prompt, max_tokens=2560, thinking_level=None):
    host = ("aiplatform.googleapis.com" if GCP_LOCATION == "global"
            else f"{GCP_LOCATION}-aiplatform.googleapis.com")
    url = (f"https://{host}/v1/projects/{GCP_PROJECT}/locations/{GCP_LOCATION}"
           f"/publishers/anthropic/models/{model_id}:rawPredict")
    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking_level:
        add_b = _CLAUDE_THINKING_BUDGET.get(thinking_level, 4096)
        payload["max_tokens"] = max_tokens + add_b
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": thinking_level}

    body = json.dumps(payload).encode()
    
    # 4 attempts max with exponential backoff
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Authorization": f"Bearer {_vertex_access_token()}",
                "Content-Type": "application/json",
            })
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=180, context=_ssl_ctx()) as r:
                d = json.loads(r.read())
            dt = time.time() - t0
            text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
            um = d.get("usage", {})
            inp, out = um.get("input_tokens", 0), um.get("output_tokens", 0)
            cr, cc = um.get("cache_read_input_tokens", 0), um.get("cache_creation_input_tokens", 0)
            
            p = PRICING.get(model_id, PRICING[SONNET_ID])
            cost = round(inp / 1e6 * p["input"] + cr / 1e6 * p["cache_read"]
                         + cc / 1e6 * p["cache_write"] + out / 1e6 * p["output"], 6)
            
            usage = {
                "input_raw": inp, "output": out, "cache_read": cr, "cache_creation": cc,
                "prompt_tokens": inp + cr + cc, "total_tokens": inp + cr + cc + out,
                "as_run_usd": cost
            }
            return text, usage, dt
        except Exception as e:
            if attempt == 3:
                raise e
            sleep_time = (2 ** attempt) + 1
            print(f"\n[Claude API Warning] {e}. Retrying in {sleep_time}s...", flush=True)
            time.sleep(sleep_time)

_gemini_client = None

def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return _gemini_client

_GEMINI_THINKING_HEADROOM = {"minimal": 2048, "low": 4096, "medium": 8192, "high": 16384}

def gemini_call(model_id, prompt, max_tokens=2560, thinking_level=None):
    from google.genai import types
    if thinking_level in ("off", "disabled"):
        thinking_level = None
    if thinking_level is None and model_id == GEMINI_FLASH_ID:
        thinking_level = "minimal"

    if thinking_level:
        tc = types.ThinkingConfig(thinking_level=thinking_level.upper())
        max_tokens = max_tokens + _GEMINI_THINKING_HEADROOM.get(thinking_level.lower(), 0)
    else:
        tc = types.ThinkingConfig(thinking_budget=0)

    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        thinking_config=tc,
        http_options=types.HttpOptions(timeout=90000)
    )

    for attempt in range(4):
        try:
            t0 = time.time()
            resp = _gemini().models.generate_content(
                model=model_id, contents=prompt, config=config
            )
            dt = time.time() - t0
            m = resp.usage_metadata
            inp = m.prompt_token_count or 0
            out = (m.candidates_token_count or 0) + (m.thoughts_token_count or 0)
            
            p = PRICING[model_id]
            cost = round(inp / 1e6 * p["input"] + out / 1e6 * p["output"], 6)
            
            usage = {
                "input_raw": inp, "output": out, "cache_read": 0, "cache_creation": 0,
                "prompt_tokens": inp, "total_tokens": m.total_token_count or 0,
                "as_run_usd": cost
            }
            return (resp.text or ""), usage, dt
        except Exception as e:
            if attempt == 3:
                raise e
            sleep_time = (2 ** attempt) + 1
            print(f"\n[Gemini API Warning] {e}. Retrying in {sleep_time}s...", flush=True)
            time.sleep(sleep_time)

def dispatch_model(model_id, prompt, max_tokens=2560, thinking_level=None):
    if model_id.startswith("gemini"):
        return gemini_call(model_id, prompt, max_tokens, thinking_level)
    return claude_api_call(model_id, prompt, max_tokens, thinking_level)
