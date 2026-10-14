"""
llm/mock_backend.py — Deterministic stand-in for a real model.

Enable with  LLM_BACKEND=mock  (or `python -m demo.token_demo --mock`).

What is REAL in mock mode
  * the flow: routing, retrieval, retries, which nodes call the LLM
  * the prompts: they are built by the real node code, so prompt sizes
    (and therefore prompt-token savings) are real (estimated at ~4 chars/token)

What is SIMULATED
  * the completions: canned text per node type, identical for every profile,
    only cut off when max_tokens is smaller than the canned answer
  → completion savings from "answer in N words" instructions are NOT modelled;
    run against Ollama/Gemini to measure those.

It also lets you exercise the critic → retry loop:  a question containing the
marker "(retry-demo)" is rejected by the mock critic exactly once.
"""

from __future__ import annotations

import json
import re

_LEARN = (
    "### Overview\n"
    "Breadth-first search (BFS) explores a graph level by level using a queue, "
    "while depth-first search (DFS) dives down one branch using a stack or recursion "
    "before backtracking. Both visit every vertex and edge once, so both run in "
    "O(V + E) time.\n\n"
    "### How they differ\n"
    "- **BFS** finds the shortest path in an unweighted graph because it reaches "
    "nodes in order of distance from the source. It needs O(V) extra space for the queue.\n"
    "- **DFS** uses O(depth) space and is the natural fit for cycle detection, "
    "topological sorting and exploring every possible path.\n\n"
    "### Example\n"
    "```python\n"
    "from collections import deque\n"
    "def bfs(graph, start):\n"
    "    seen, q = {start}, deque([start])\n"
    "    while q:\n"
    "        node = q.popleft()\n"
    "        for nxt in graph[node]:\n"
    "            if nxt not in seen:\n"
    "                seen.add(nxt); q.append(nxt)\n"
    "```\n\n"
    "Swap the deque for a stack (or recurse) and you have DFS.\n\n"
    "**Think about it:** which of the two would you pick to find the fewest "
    "hops between two users in a social network, and why?"
)

_HINT = (
    "### Hint (Level 1)\n"
    "Think about what you already know when you look at one number: if you are "
    "standing on `nums[i]`, exactly which value would complete the target? "
    "Is there a data structure that lets you ask \"have I seen that value before?\" "
    "in constant time instead of re-scanning the array?"
)

_SOLUTION = (
    "### Solution (Python)\n"
    "```python\n"
    "def two_sum(nums, target):\n"
    "    seen = {}                      # value -> index\n"
    "    for i, x in enumerate(nums):\n"
    "        need = target - x\n"
    "        if need in seen:           # complement already visited\n"
    "            return [seen[need], i]\n"
    "        seen[x] = i\n"
    "    return []\n"
    "```\n\n"
    "**Complexity:** O(n) time, O(n) space.\n\n"
    "**Why it works:** one pass; for each element we check whether its complement "
    "was already stored in the hash map, trading memory for speed.\n\n"
    "**Edge cases:** duplicate values (`[3, 3]`, target 6), negative numbers, "
    "no valid pair, and reusing the same index twice."
)

_CODE_REVIEW = {
    "intro": "Nice start - your brute-force approach is correct and easy to follow.",
    "analysis": "Two nested loops try every pair of indices and return the first pair "
                "that sums to the target, so the logic is sound for all valid inputs.",
    "complexity": "Time O(n^2) because of the nested loops; space O(1) as no extra "
                  "structure is allocated.",
    "evaluation": "Correct for standard cases and duplicates; returns an empty list when "
                  "there is no pair, which is a reasonable choice for the edge case.",
    "hint": "Ask yourself whether you can remember earlier numbers to avoid the inner loop.",
    "feedback": "1) Store seen values in a hash map for O(n). 2) Name variables "
                "more descriptively than i and j. 3) Add a short docstring.",
    "encouragement": "Great foundation - one small idea turns this into an optimal answer!",
    "scorecard": {"Correctness": 4, "Efficiency": 1, "Readability": 1, "Approach": 1},
}


def _between(prompt: str, start: str, end: str | None = None) -> str:
    i = prompt.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = prompt.find(end, i) if end else -1
    return prompt[i:j if j >= 0 else None].strip()


def _route(prompt: str) -> str:
    msg = (_between(prompt, "Student message:", "Return ONLY")
           or _between(prompt, "Message:", "Answer:")).lower()
    for intent, words in (("code_review", ("review", "debug", "my code")),
                          ("hint", ("hint", "stuck")),
                          ("solution", ("solution",)),
                          ("practice", ("problem", "practice")),
                          ("direct", ("hello", "thanks"))):
        if any(w in msg for w in words):
            return intent
    return "learn"


_retry_seen: set[str] = set()


def reset() -> None:
    _retry_seen.clear()


def _critic(prompt: str) -> str:
    question = _between(prompt, "Student question:", "Draft response:")
    if "(retry-demo)" in question and "[Improve based on this critique" not in question:
        if question not in _retry_seen:
            _retry_seen.add(question)
            return json.dumps({"verdict": "bad",
                               "critique": "Add a concrete example and state the complexity."})
    return json.dumps({"verdict": "good", "critique": ""})


def complete(prompt: str, *, max_tokens: int | None = None) -> str:
    """Return canned text for the node that produced *prompt*."""
    if "routing system for a DSA AI Coach" in prompt or "Classify into one word" in prompt:
        text = _route(prompt)
    elif "strict quality reviewer" in prompt:
        text = _critic(prompt)
    elif "DSA Code Review Agent" in prompt:
        text = json.dumps(_CODE_REVIEW, indent=2)
    elif "Hint Agent" in prompt:
        text = _HINT
    elif "revealing the optimal solution" in prompt:
        text = _SOLUTION
    else:
        text = _LEARN

    if max_tokens:                       # emulate hard truncation at num_predict
        text = text[: max_tokens * 4]
    return text
