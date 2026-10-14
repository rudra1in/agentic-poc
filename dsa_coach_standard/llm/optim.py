"""
llm/optim.py — Token-optimisation switches (one flag per technique).

Each flag maps to one numbered technique in the README:

  T1 lean_router       keyword-first routing, tiny fallback prompt, 8-token answer
  T2 lazy_retrieval    skip RAG for intents that never read the context
  T3 context_budget    smaller top-k, relevance cut-off, char budget for RAG + code
  T4 history_window    last 4 turns, each clipped to 240 chars
  T5 output_caps       per-node max_tokens + "answer in N words" instructions
  T6 selective_critic  LLM critic only for learn/solution, on a shorter draft
  T7 retry_direct      a critic retry goes straight back to the agent
  T8 response_cache    exact-match cache for identical prompts
  T9 model_tiering     router + critic use the small model tier (if configured)

Profiles
  OPT_PROFILE=optimized (default) → every flag on
  OPT_PROFILE=baseline            → every flag off  (original behaviour)
  OPT_DISABLE=response_cache,selective_critic   → switch single flags off
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass
class Flags:
    lean_router: bool = False
    lazy_retrieval: bool = False
    context_budget: bool = False
    history_window: bool = False
    output_caps: bool = False
    selective_critic: bool = False
    retry_direct: bool = False
    response_cache: bool = False
    model_tiering: bool = False


FLAG_NAMES: list[str] = [f.name for f in fields(Flags)]

TECHNIQUES: dict[str, str] = {
    "lean_router":      "T1  Lean router",
    "lazy_retrieval":   "T2  Lazy retrieval",
    "context_budget":   "T3  Context budget",
    "history_window":   "T4  History window",
    "output_caps":      "T5  Output caps",
    "selective_critic": "T6  Selective critic",
    "retry_direct":     "T7  Direct retry",
    "response_cache":   "T8  Response cache",
    "model_tiering":    "T9  Model tiering",
}


def _from_env() -> Flags:
    on = os.getenv("OPT_PROFILE", "optimized").lower() == "optimized"
    flags = Flags(**{name: on for name in FLAG_NAMES})
    for name in os.getenv("OPT_DISABLE", "").split(","):
        name = name.strip()
        if name in FLAG_NAMES:
            setattr(flags, name, False)
    return flags


# Global, mutable on purpose — the demo flips flags at runtime.
F: Flags = _from_env()


def set_profile(profile: str) -> None:
    on = profile.lower() == "optimized"
    for name in FLAG_NAMES:
        setattr(F, name, on)


def set_only(enabled: list[str]) -> None:
    """Enable exactly the given flags (used by the cumulative ablation)."""
    for name in FLAG_NAMES:
        setattr(F, name, name in enabled)


# ----------------------------------------------------------------
# Budgets
# ----------------------------------------------------------------
NO_CONTEXT_INTENTS = {"practice", "direct"}   # never read state["context"]
CRITIC_INTENTS     = {"learn", "solution"}    # free-form prose worth reviewing

RAG_TOP_K_BASE, RAG_TOP_K_OPT = 5, 3
RAG_MAX_CHARS                 = 700           # ≈ 175 tokens of retrieved context
HISTORY_TURNS_BASE            = 6             # original behaviour
HISTORY_TURNS_OPT             = 4
HISTORY_CHARS_OPT             = 240
CODE_MAX_CHARS                = 4000
CRITIC_DRAFT_CHARS_BASE       = 1500          # original behaviour
CRITIC_DRAFT_CHARS_OPT        = 800
CRITIQUE_MAX_CHARS            = 200

CODE_REVIEW_MAX_TOKENS        = 500           # full 8-field JSON needs ~350–450

CAPS = {"router": 8, "critic": 80, "learn": 500, "hint": 220, "solution": 700}


def cap(tag: str) -> int | None:
    """Output-token cap for a node (None = client default)."""
    return CAPS.get(tag) if F.output_caps else None


def brief(words: int) -> str:
    """Brevity instruction appended to prompts when caps are on."""
    return f"\nKeep the whole answer under {words} words." if F.output_caps else ""


def tier() -> str:
    return "small" if F.model_tiering else "main"


def rag_top_k() -> int:
    return RAG_TOP_K_OPT if F.context_budget else RAG_TOP_K_BASE


def critic_draft_chars() -> int:
    return CRITIC_DRAFT_CHARS_OPT if F.selective_critic else CRITIC_DRAFT_CHARS_BASE


# ----------------------------------------------------------------
# Prompt-side helpers
# ----------------------------------------------------------------
def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())            # collapse whitespace / newlines
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def window(history: list[dict]) -> list[dict]:
    """Slice of conversation history that goes into a prompt."""
    if not F.history_window:
        return history[-HISTORY_TURNS_BASE:]
    return [
        {**m, "content": _clip(m.get("content", ""), HISTORY_CHARS_OPT)}
        for m in history[-HISTORY_TURNS_OPT:]
    ]


def fit_chunks(chunks: list[str], max_chars: int = RAG_MAX_CHARS) -> list[str]:
    """Drop duplicate chunks and stop adding once the char budget is used."""
    seen, out, used = set(), [], 0
    for c in chunks:
        key = " ".join(c.lower().split())
        if key in seen:
            continue
        if used + len(c) > max_chars:
            break
        seen.add(key)
        out.append(c)
        used += len(c)
    return out


def cap_code(code: str) -> str:
    """Keep head + tail of very long submissions."""
    if not F.context_budget or len(code) <= CODE_MAX_CHARS:
        return code
    half = CODE_MAX_CHARS // 2
    return code[:half] + "\n# … (middle of file omitted to save tokens) …\n" + code[-half:]
