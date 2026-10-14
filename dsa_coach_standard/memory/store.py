"""
memory/store.py — Progress and conversation persistence

Merges:
  - root CodeCurry  : progress.json (score history, solved IDs)
  - AURA++          : per-session conversation + problem progress
"""

import json
from pathlib import Path
from typing import Any

BASE_DIR      = Path(__file__).resolve().parents[1]
PROGRESS_FILE = BASE_DIR / "data" / "progress.json"
CONV_DIR      = BASE_DIR / "data" / "conversations"

CONV_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_PROGRESS = {
    "score_history": [],
    "avg_score":     0.0,
    "solved_ids":    [],
    "submissions":   0,
}


# ----------------------------------------------------------------
# Score / progress
# ----------------------------------------------------------------

def load_progress() -> dict[str, Any]:
    if not PROGRESS_FILE.exists():
        return DEFAULT_PROGRESS.copy()
    try:
        data   = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        scores = [max(0, min(10, int(float(x)))) for x in data.get("score_history", [])
                  if str(x).replace(".", "", 1).isdigit()]
        solved = data.get("solved_ids", [])
        avg    = sum(scores) / len(scores) if scores else 0.0
        return {"score_history": scores, "avg_score": avg,
                "solved_ids": solved, "submissions": len(scores)}
    except Exception:
        return DEFAULT_PROGRESS.copy()


def save_progress(score_history: list[int], avg_score: float, problem_id: str = "") -> None:
    clean = [max(0, min(10, int(float(x)))) for x in score_history
             if str(x).replace(".", "", 1).isdigit()]
    existing = load_progress()
    solved   = existing.get("solved_ids", [])
    if problem_id and problem_id not in solved:
        solved.append(problem_id)
    data = {
        "score_history": clean,
        "avg_score":     sum(clean) / len(clean) if clean else 0.0,
        "solved_ids":    solved,
        "submissions":   len(clean),
    }
    PROGRESS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ----------------------------------------------------------------
# Conversation history (per session)
# ----------------------------------------------------------------

def _conv_path(session_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return CONV_DIR / f"{safe}.json"


def load_conversation(session_id: str, limit: int = 20) -> list[dict]:
    path = _conv_path(session_id)
    if not path.exists():
        return []
    try:
        messages = json.loads(path.read_text(encoding="utf-8"))
        return messages[-limit:]
    except Exception:
        return []


def save_message(session_id: str, role: str, content: str, **meta: Any) -> None:
    """Append a message. Extra keyword args (e.g. problem_id) are stored alongside."""
    path     = _conv_path(session_id)
    messages = load_conversation(session_id, limit=200)
    messages.append({"role": role, "content": content, **meta})
    path.write_text(json.dumps(messages, indent=2), encoding="utf-8")


def clear_conversation(session_id: str) -> None:
    _conv_path(session_id).unlink(missing_ok=True)
