"""
agents/router.py — Unified intent router

Strategy (in priority order):
  1. Explicit sidebar `mode` field  ← lumina                (free)
  2. Keyword rules                  ← lumina / root         (free)
  3. LLM classification             ← AURA++                (costs tokens)

With the `lean_router` flag (llm/optim.py) step 3 only runs when NO keyword
matched, uses a ~40-token prompt and an 8-token answer.  Without it the
original behaviour is kept: a "learn" keyword hit is still double-checked by
the LLM.

Intents / agent_types:
  learn       — explain a DSA concept via RAG
  practice    — fetch a problem from the problem bank
  hint        — give a hint for the current problem
  solution    — reveal the full solution
  code_review — analyse student code and give feedback
  direct      — greeting / off-topic (no tool needed)
"""

import re

from agents.state import AgentState
from llm.client import chat
from llm import optim

# ----------------------------------------------------------------
# Sidebar mode → intent map  (lumina)
# ----------------------------------------------------------------
_MODE_MAP: dict[str, str] = {
    "learn": "learn",
    "learn dsa": "learn",
    "practice": "practice",
    "hint": "hint",
    "get hint": "hint",
    "solution": "solution",
    "view solution": "solution",
    "code_review": "code_review",
    "code review": "code_review",
    "debug": "code_review",
}

# ----------------------------------------------------------------
# Keyword → intent map  (lumina / root fallback)
# ----------------------------------------------------------------
_KEYWORD_INTENTS: list[tuple[str, list[str]]] = [
    ("code_review", ["review", "debug", "error", "wrong answer", "bug", "optimize",
                     "runtime error", "tle", "my code", "my solution", "why does my code",
                     "compile error", "time complexity of my code"]),
    ("hint",        ["hint", "clue", "stuck", "nudge", "give me a hint"]),
    ("solution",    ["solution", "solve", "answer", "complete code", "full solution"]),
    ("practice",    ["practice", "leetcode", "coding problem", "give me a problem",
                     "easy problem", "medium problem", "hard problem", "dp problem",
                     "array problem", "tree problem"]),
    ("learn",       ["explain", "what is", "how does", "teach me", "concept",
                     "difference between", "complexity", "algorithm"]),
    ("direct",      ["hello", "hi", "hey", "thanks", "thank you"]),
]


def _compile(kw: str) -> "re.Pattern[str]":
    # whole-word match (so "hi" no longer fires on "which"/"high"), simple plurals allowed
    plural = r"(?:s|es)?" if len(kw) > 3 else ""
    return re.compile(rf"\b{re.escape(kw)}{plural}\b")


_KEYWORD_RES = [(intent, [_compile(k) for k in kws]) for intent, kws in _KEYWORD_INTENTS]


def _keyword_intent(question: str) -> tuple[str, bool]:
    """Return (intent, matched).  matched=False means "no rule fired" (default learn)."""
    q = question.lower()
    for intent, patterns in _KEYWORD_RES:
        if any(p.search(q) for p in patterns):
            return intent, True
    return "learn", False


_VALID_INTENTS = ("learn", "practice", "hint", "solution", "code_review", "direct")

_FULL_PROMPT = """You are the routing system for a DSA AI Coach.

Classify the student's message into exactly ONE of these intents:

learn       — student wants to understand a DSA concept or algorithm
practice    — student wants a DSA problem to solve
hint        — student wants a hint (not the full solution)
solution    — student wants the complete solution
code_review — student submitted code for feedback / debug / analysis
direct      — greeting, small talk, or off-topic

Student message:
{question}

Return ONLY one word (the intent). No explanation."""

# T1: same task in ~1/4 of the tokens
_LEAN_PROMPT = """Classify into one word: learn|practice|hint|solution|code_review|direct
Message: {question}
Answer:"""


def _llm_intent(question: str) -> str:
    """Use LLM to classify intent when keyword matching is ambiguous."""
    if optim.F.lean_router:
        prompt = _LEAN_PROMPT.format(question=question[:500])
        raw = chat(prompt, tier=optim.tier(), max_tokens=optim.cap("router") or 8,
                   temperature=0, tag="router")
    else:
        prompt = _FULL_PROMPT.format(question=question)
        raw = chat(prompt, tag="router")

    raw = raw.strip().lower().replace(" ", "_")
    hits = [(raw.find(i), i) for i in _VALID_INTENTS if i in raw]
    return min(hits)[1] if hits else "learn"


# ----------------------------------------------------------------
# LangGraph node
# ----------------------------------------------------------------

def router_node(state: AgentState) -> AgentState:
    """
    Determines `intent` and `agent_type` for the current request.
    Sets `route` (AURA++ compat) as well.
    """
    question = str(state.get("question", "")).strip()
    mode = str(state.get("mode", "")).lower().strip()

    # 1. Sidebar mode wins
    if mode in _MODE_MAP:
        intent = _MODE_MAP[mode]
        print(f"[ROUTER] Sidebar mode '{mode}' → intent: {intent}")
    else:
        kw_intent, matched = _keyword_intent(question)
        if optim.F.lean_router:
            # 2. Any keyword hit is trusted (free). LLM only when nothing matched.
            if matched or not question:
                intent = kw_intent
                print(f"[ROUTER] Keyword match → intent: {intent}")
            else:
                intent = _llm_intent(question)
                print(f"[ROUTER] LLM classified → intent: {intent}")
        else:
            # Original behaviour: even a "learn" keyword hit is double-checked by the LLM
            if kw_intent != "learn" or not question:
                intent = kw_intent
                print(f"[ROUTER] Keyword match → intent: {intent}")
            else:
                intent = _llm_intent(question)
                print(f"[ROUTER] LLM classified → intent: {intent}")

    # Map intent → agent_type (AgentOps compat)
    agent_type_map = {
        "learn":       "coach",
        "practice":    "coach",
        "hint":        "hint",
        "solution":    "coach",
        "code_review": "code",
        "direct":      "direct",
    }
    agent_type = agent_type_map.get(intent, "coach")

    return {
        **state,
        "intent":     intent,
        "agent_type": agent_type,
        "route":      intent.upper(),   # AURA++ compat field
    }
