"""
llm/client.py — Unified LLM client

Supports:
  - Ollama  (local, default)   ← root CodeCurry / AURA++ / lumina
  - Gemini  (cloud fallback)   ← lumina uses google-genai
  - mock    (offline demo/tests, see llm/mock_backend.py)

Set LLM_BACKEND env var to "gemini" or "mock" to switch.

Every call is recorded in ``llm.usage.tracker`` (tokens, latency, node tag) so
you can see where the budget goes, and — when the ``response_cache`` flag in
``llm.optim`` is on — identical prompts are answered from memory for 0 tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

try:                                   # .env was documented but never loaded before
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from llm import optim
from llm.usage import CallRecord, estimate_tokens, tracker

LLM_BACKEND          = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_PLANNER_MODEL = os.getenv("OLLAMA_PLANNER_MODEL", "qwen2.5-coder:7b")
# Optional cheaper tier for routing / critic.  Defaults to the main model, so
# turning on `model_tiering` changes nothing until you set e.g. llama3.2:1b.
OLLAMA_SMALL_MODEL   = os.getenv("OLLAMA_SMALL_MODEL", "") or OLLAMA_MODEL
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_SMALL_MODEL   = os.getenv("GEMINI_SMALL_MODEL", "") or GEMINI_MODEL

# "api"      → use the backend's own counters (fallback to estimate if missing)
# "estimate" → prompt tokens = chars/4 for every backend (avoids KV-cache skew)
TOKEN_COUNT_MODE     = os.getenv("TOKEN_COUNT_MODE", "api").lower()

DEFAULT_CHAT_TOKENS  = 1200            # original num_predict for prose
DEFAULT_JSON_TOKENS  = 120             # original num_predict for JSON helpers


# ----------------------------------------------------------------
# Response cache (technique T8)
# ----------------------------------------------------------------
_CACHE: dict[str, tuple[str, int, int]] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _cache_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()


# ----------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------
def model_name(tier: str = "main", planner: bool = False) -> str:
    if LLM_BACKEND == "mock":
        return "mock-small" if tier == "small" else "mock-main"
    if LLM_BACKEND == "gemini":
        return GEMINI_SMALL_MODEL if tier == "small" else GEMINI_MODEL
    if planner:
        return OLLAMA_PLANNER_MODEL
    return OLLAMA_SMALL_MODEL if tier == "small" else OLLAMA_MODEL


# ----------------------------------------------------------------
# Backends — each returns (text, prompt_tokens|None, completion_tokens|None)
# ----------------------------------------------------------------
def _field(resp: Any, name: str) -> int | None:
    try:
        v = resp[name]
    except Exception:
        v = getattr(resp, name, None)
    return int(v) if isinstance(v, (int, float)) else None


def _ollama(prompt: str, model: str, json_mode: bool, max_tokens: int,
            temperature: float | None) -> tuple[str, int | None, int | None]:
    import ollama

    messages: list[dict] = []
    options: dict[str, Any] = {"num_predict": max_tokens}
    kwargs: dict[str, Any] = {}
    if json_mode:
        messages.append({"role": "system",
                         "content": "You are a strict JSON responder. Return only valid JSON."})
        kwargs["format"] = "json"
        options["temperature"] = 0
    if temperature is not None:
        options["temperature"] = temperature
    messages.append({"role": "user", "content": prompt})

    resp = ollama.chat(model=model, messages=messages, options=options, **kwargs)
    return (resp["message"]["content"],
            _field(resp, "prompt_eval_count"),
            _field(resp, "eval_count"))


def _gemini(prompt: str, model: str) -> tuple[str, int | None, int | None]:
    # NOTE: no max_output_tokens here — on 2.5 models "thinking" tokens share that
    # budget, so a tight cap can return an empty answer. Brevity is enforced in
    # the prompt instead (optim.brief).
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(model=model, contents=prompt)
    um = getattr(resp, "usage_metadata", None)
    return (resp.text or "",
            getattr(um, "prompt_token_count", None),
            getattr(um, "candidates_token_count", None))


def _mock(prompt: str, max_tokens: int) -> tuple[str, int | None, int | None]:
    from llm import mock_backend
    return mock_backend.complete(prompt, max_tokens=max_tokens), None, None


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------

def chat(
    prompt: str,
    *,
    json_mode: bool = False,
    planner: bool = False,
    tier: str = "main",
    max_tokens: int | None = None,
    temperature: float | None = None,
    tag: str = "llm",
) -> str:
    """
    Send a prompt to the configured LLM backend.

    Args:
        prompt:      The full prompt string.
        json_mode:   If True, instruct the model to return pure JSON.
        planner:     Use OLLAMA_PLANNER_MODEL (legacy switch, Ollama only).
        tier:        "main" or "small" (small = OLLAMA_SMALL_MODEL / GEMINI_SMALL_MODEL).
        max_tokens:  Output cap (Ollama num_predict). Default 1200 prose / 120 JSON.
        temperature: Optional sampling temperature (Ollama).
        tag:         Node name used for the usage report.

    Returns:
        Raw text response from the LLM ("" on backend error).
    """
    model = model_name(tier, planner)
    limit = max_tokens or (DEFAULT_JSON_TOKENS if json_mode else DEFAULT_CHAT_TOKENS)

    key = None
    if optim.F.response_cache:
        key = _cache_key(LLM_BACKEND, model, json_mode, limit, temperature, prompt)
        hit = _CACHE.get(key)
        if hit:
            text, pt, ct = hit
            tracker.add(CallRecord(tag, model, 0, 0, 0.0, cached=True, saved_tokens=pt + ct))
            print(f"[LLM] {tag}: cache hit (saved ~{pt + ct} tokens)")
            return text

    t0 = time.perf_counter()
    try:
        if LLM_BACKEND == "mock":
            text, pt, ct = _mock(prompt, limit)
        elif LLM_BACKEND == "gemini":
            text, pt, ct = _gemini(prompt, model)
        else:
            text, pt, ct = _ollama(prompt, model, json_mode, limit, temperature)
    except Exception as exc:
        print(f"[LLM] Backend '{LLM_BACKEND}' error: {exc}")
        return ""
    latency = time.perf_counter() - t0

    estimated = pt is None or ct is None or TOKEN_COUNT_MODE == "estimate" or LLM_BACKEND == "mock"
    if pt is None or TOKEN_COUNT_MODE == "estimate" or LLM_BACKEND == "mock":
        pt = estimate_tokens(prompt) + (12 if json_mode else 0)
    if ct is None:
        ct = estimate_tokens(text)

    tracker.add(CallRecord(tag, model, pt, ct, latency, estimated=estimated))
    if key and text:
        _CACHE[key] = (text, pt, ct)
    return text


# ----------------------------------------------------------------
# JSON extraction utility (used across all nodes)
# ----------------------------------------------------------------

def extract_json(text: str) -> dict[str, Any]:
    """Robustly parse JSON from LLM output, stripping markdown fences."""
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "")
    try:
        return json.loads(cleaned.strip())
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def clamp(value: Any, low: int, high: int) -> int:
    """Clamp a numeric value to [low, high]."""
    try:
        return max(low, min(high, int(float(value))))
    except Exception:
        return low
