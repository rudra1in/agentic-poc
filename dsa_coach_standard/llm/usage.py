"""
llm/usage.py — Token & call accounting for every LLM request.

Every call made through ``llm.client.chat`` is recorded here, tagged with the
node that made it ("router", "learn", "critic", ...).  The demo reads these
records to build the before/after report; the app can also print them:

    from llm.usage import tracker
    print(tracker.totals())
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token for English / code)."""
    return max(1, len(text) // 4) if text else 0


@dataclass
class CallRecord:
    tag: str                    # node that made the call (router, learn, critic …)
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool = False        # served from the response cache (0 tokens spent)
    estimated: bool = False     # True when the backend gave no token counts
    saved_tokens: int = 0       # tokens NOT spent thanks to a cache hit

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[CallRecord] = []

    def add(self, rec: CallRecord) -> None:
        with self._lock:
            self.records.append(rec)

    def reset(self) -> None:
        with self._lock:
            self.records.clear()

    def mark(self) -> int:
        """Index to slice from later:  tracker.records[mark:]"""
        return len(self.records)

    # ── aggregation helpers ──────────────────────────────────────
    @staticmethod
    def totals_for(records: list[CallRecord]) -> dict:
        real = [r for r in records if not r.cached]
        return {
            "calls":             len(real),
            "cache_hits":        sum(1 for r in records if r.cached),
            "prompt_tokens":     sum(r.prompt_tokens for r in real),
            "completion_tokens": sum(r.completion_tokens for r in real),
            "total_tokens":      sum(r.total for r in real),
            "saved_by_cache":    sum(r.saved_tokens for r in records if r.cached),
            "latency_s":         round(sum(r.latency_s for r in real), 2),
        }

    def totals(self) -> dict:
        return self.totals_for(self.records)

    @staticmethod
    def by_tag_for(records: list[CallRecord]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in records:
            if r.cached:
                continue
            d = out.setdefault(r.tag, {"calls": 0, "prompt": 0, "completion": 0})
            d["calls"] += 1
            d["prompt"] += r.prompt_tokens
            d["completion"] += r.completion_tokens
        return out

    def by_tag(self) -> dict[str, dict]:
        return self.by_tag_for(self.records)


tracker = UsageTracker()
