"""
rag/retriever.py — RAG retrieval layer

Priority:
  1. pgvector (PostgreSQL)         ← root CodeCurry / AURA++
  2. ChromaDB                      ← lumina / AgentOps
  3. In-memory keyword fallback    ← always available

Set RAG_BACKEND env var: "pgvector" | "chroma" | "memory" (default)
"""

from __future__ import annotations
import math
import os
import re
from typing import Any

from llm import optim

RAG_BACKEND = os.getenv("RAG_BACKEND", "memory").lower()

# ----------------------------------------------------------------
# pgvector backend
# ----------------------------------------------------------------

def _retrieve_pgvector(query: str, top_k: int) -> list[dict]:
    import psycopg2
    import numpy as np
    from llm.client import chat

    DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost/dsa_coach")
    embed_prompt = f"Represent this for retrieval:\n{query}"
    # Naive: ask LLM to summarise query; replace with real embedding model
    conn   = psycopg2.connect(DB_URL)
    cur    = conn.cursor()
    cur.execute(
        "SELECT content FROM knowledge_chunks ORDER BY embedding <-> %s LIMIT %s",
        (query, top_k),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"text": r[0]} for r in rows]


# ----------------------------------------------------------------
# ChromaDB backend
# ----------------------------------------------------------------

def _retrieve_chroma(query: str, top_k: int) -> list[dict]:
    import chromadb
    from chromadb.utils import embedding_functions

    client     = chromadb.Client()
    ef         = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection("dsa_knowledge", embedding_function=ef)
    results    = collection.query(query_texts=[query], n_results=top_k)
    docs       = results.get("documents", [[]])[0]
    return [{"text": d} for d in docs]


# ----------------------------------------------------------------
# In-memory keyword fallback
# ----------------------------------------------------------------

_KNOWLEDGE_BASE: list[dict] = [
    {"text": "Dynamic programming solves problems by breaking them into overlapping subproblems and storing results (memoization or tabulation). Time: O(n²) or better. Space: O(n)."},
    {"text": "Two-pointer technique uses two indices moving toward each other or in same direction to solve array/string problems in O(n) instead of O(n²)."},
    {"text": "BFS uses a queue to explore nodes level by level. Time: O(V+E). Used for shortest path in unweighted graphs."},
    {"text": "DFS uses a stack (or recursion) to explore as far as possible before backtracking. Time: O(V+E). Used for cycle detection, topological sort."},
    {"text": "Binary search finds an element in a sorted array in O(log n). Template: lo=0, hi=n-1; while lo<=hi: mid=(lo+hi)//2."},
    {"text": "Sliding window maintains a contiguous subarray/substring. Expand right, contract left when condition violated. O(n)."},
    {"text": "Hash map / dictionary provides O(1) average lookup. Excellent for frequency counting, Two Sum, anagram detection."},
    {"text": "Heap (priority queue) gives O(log n) insert/delete and O(1) peek of min/max. Used for K-th largest, Dijkstra."},
    {"text": "Backtracking explores all possibilities by building solutions incrementally and abandoning invalid paths early. Examples: N-Queens, Sudoku, subsets."},
    {"text": "Greedy algorithms make locally optimal choices hoping for global optimum. Works for interval scheduling, Huffman coding, coin change (certain coin systems)."},
    {"text": "Tree traversals: In-order (left-root-right) gives sorted BST. Pre-order (root-left-right) for tree copy. Post-order (left-right-root) for deletion."},
    {"text": "Union-Find (Disjoint Set Union) detects cycles and groups connected components in near O(1) with path compression + union by rank."},
    {"text": "Trie (prefix tree) stores strings efficiently for prefix search. Insert/Search: O(m) where m = word length."},
    {"text": "Monotonic stack maintains elements in increasing or decreasing order. Used for next greater element, largest rectangle in histogram."},
]


_STOPWORDS = {
    "the", "and", "for", "with", "what", "how", "does", "did", "can", "you", "are",
    "this", "that", "from", "into", "about", "explain", "give", "show", "tell", "please",
    "vs", "versus", "example", "examples", "between", "difference", "why", "when", "use", "using",
}


def _norm_words(text: str) -> set[str]:
    words = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) < 3 or w in _STOPWORDS:
            continue
        words.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return words


def _retrieve_memory_scored(query: str, top_k: int) -> list[dict]:
    """
    T3: content-word overlap weighted by rarity (idf), with a relevance cut-off.
    Returns FEWER than top_k chunks when little is relevant — every chunk that is
    not sent to the model is a chunk you do not pay for.
    """
    q_words = _norm_words(query)
    chunk_words = [_norm_words(c["text"]) for c in _KNOWLEDGE_BASE]
    n = len(_KNOWLEDGE_BASE)
    df = {w: sum(w in cw for cw in chunk_words) for w in q_words}
    idf = {w: math.log((n + 1) / (df[w] + 1)) + 1 for w in q_words}

    scored = [(sum(idf[w] for w in q_words if w in cw), c)
              for cw, c in zip(chunk_words, _KNOWLEDGE_BASE)]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][0] if scored else 0
    if best <= 0:
        return []
    return [c for s, c in scored[:top_k] if s >= 0.4 * best]


def _retrieve_memory(query: str, top_k: int) -> list[dict]:
    if optim.F.context_budget:
        return _retrieve_memory_scored(query, top_k)
    # original scoring (substring hits — short words like "a"/"of" match everything)
    q = query.lower()
    scored = [(sum(w in q for w in chunk["text"].lower().split()), chunk)
              for chunk in _KNOWLEDGE_BASE]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------

def retrieve_chunks(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve relevant DSA knowledge chunks for a query.
    Returns a list of dicts with at least a "text" key.
    """
    if RAG_BACKEND == "pgvector":
        return _retrieve_pgvector(query, top_k)
    if RAG_BACKEND == "chroma":
        return _retrieve_chroma(query, top_k)
    return _retrieve_memory(query, top_k)
