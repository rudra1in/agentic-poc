"""
agents/graph.py — Standard DSA Coach LangGraph

Architecture merges:
  - Root CodeCurry  : linear retrieve → coach → progress pipeline
  - AURA++          : ReAct router with conditional edges per intent
  - lumina          : critic → retry loop, multi-agent specialization
  - AgentOps        : evaluation node, checkpointer, memory node

Final graph flow:
  START
    ↓
  memory          ← load history + progress
    ↓
  router          ← classify intent → agent_type
    ↓
  retrieve        ← RAG (optional, graceful skip; lazy for practice/direct)
    ↓
  ┌────────────────────────────────┐
  │  learn / practice / hint /     │
  │  solution / code_review /      │
  │  direct                        │
  └────────────────────────────────┘
    ↓
  critic          ← quality check draft_response
    ↓
  retry? ── yes ──→  agent again (critique appended; router skipped when
                     `retry_direct` is on, otherwise back through router)
    ↓ no
  motivation      ← add encouragement (code review only)
    ↓
  final           ← assemble final_response
    ↓
  END
"""

from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.router import router_node
from llm import optim
from agents.nodes import (
    memory_node,
    retrieve_node,
    learn_node,
    practice_node,
    hint_node,
    solution_node,
    code_review_node,
    critic_node,
    motivation_node,
    direct_node,
    final_node,
)


# ----------------------------------------------------------------
# Conditional edge functions
# ----------------------------------------------------------------

def _choose_agent(state: AgentState) -> str:
    """Route to the correct specialized agent after RAG retrieval."""
    intent = state.get("intent", "learn")
    mapping = {
        "learn":       "learn",
        "practice":    "practice",
        "hint":        "hint",
        "solution":    "solution",
        "code_review": "code_review",
        "direct":      "direct",
    }
    choice = mapping.get(intent, "learn")
    print(f"[GRAPH] Intent '{intent}' → node '{choice}'")
    return choice


def _after_critic(state: AgentState) -> str:
    """
    If the critic flags a bad response AND retries remain, loop back.
    Otherwise proceed to motivation / final.
    """
    needs_retry = state.get("needs_retry", False)
    loop_count  = state.get("loop_count", 0)
    max_retries = state.get("max_retries", 2)

    if needs_retry and loop_count < max_retries:
        print(f"[GRAPH] Critic requested retry (loop {loop_count + 1}/{max_retries})")
        return "retry"
    return "proceed"


def _after_motivation(state: AgentState) -> str:
    """Only apply motivation node for code_review intent."""
    return "final"


def _prepare_retry(state: AgentState) -> AgentState:
    """Append critique to question so the agent improves on next pass."""
    critique   = state.get("critique", "")
    if optim.F.retry_direct:                 # T7: keep the appended critique short
        critique = critique[: optim.CRITIQUE_MAX_CHARS]
    question   = state.get("question", "")
    loop_count = state.get("loop_count", 0)
    return {
        **state,
        "question":   question + f"\n\n[Improve based on this critique: {critique}]",
        "loop_count": loop_count + 1,
        "draft_response": "",
    }


def _after_retry(state: AgentState) -> str:
    """
    T7: the intent was already decided on the first pass, so with `retry_direct`
    the retry goes straight back to the same agent (context is still in state).
    Original behaviour: back through router (re-classification, maybe an LLM call).
    """
    if optim.F.retry_direct:
        return _choose_agent(state)
    return "router"


# ----------------------------------------------------------------
# Build graph
# ----------------------------------------------------------------

def build_graph(use_checkpointer: bool = False) -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────
    graph.add_node("memory",      memory_node)
    graph.add_node("retrieve",    retrieve_node)
    graph.add_node("router",      router_node)

    graph.add_node("learn",       learn_node)
    graph.add_node("practice",    practice_node)
    graph.add_node("hint",        hint_node)
    graph.add_node("solution",    solution_node)
    graph.add_node("code_review", code_review_node)
    graph.add_node("direct",      direct_node)

    graph.add_node("critic",      critic_node)
    graph.add_node("retry",       _prepare_retry)
    graph.add_node("motivation",  motivation_node)
    graph.add_node("final",       final_node)

    # ── Edges ─────────────────────────────────────────────────────

    # Entry pipeline  (router first, so retrieval can depend on the intent)
    graph.add_edge(START,      "memory")
    graph.add_edge("memory",   "router")
    graph.add_edge("router",   "retrieve")

    # Retrieve → specialized agent (conditional)
    graph.add_conditional_edges(
        "retrieve",
        _choose_agent,
        {
            "learn":       "learn",
            "practice":    "practice",
            "hint":        "hint",
            "solution":    "solution",
            "code_review": "code_review",
            "direct":      "direct",
        },
    )

    # All agents → critic
    for agent in ("learn", "practice", "hint", "solution", "code_review"):
        graph.add_edge(agent, "critic")

    # direct → skip critic → final directly
    graph.add_edge("direct", "final")

    # Critic → retry loop or proceed
    graph.add_conditional_edges(
        "critic",
        _after_critic,
        {
            "retry":   "retry",
            "proceed": "motivation",
        },
    )

    # Retry → router (original) or straight back to the agent (retry_direct)
    graph.add_conditional_edges(
        "retry",
        _after_retry,
        {
            "router":      "router",
            "learn":       "learn",
            "practice":    "practice",
            "hint":        "hint",
            "solution":    "solution",
            "code_review": "code_review",
            "direct":      "direct",
        },
    )

    # Motivation → final
    graph.add_edge("motivation", "final")

    # Final → END
    graph.add_edge("final", END)

    # ── Compile ───────────────────────────────────────────────────
    if use_checkpointer:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            return graph.compile(checkpointer=checkpointer)
        except Exception as exc:
            print(f"[GRAPH] Checkpointer unavailable ({exc}); compiling without.")

    return graph.compile()


# ── Module-level compiled graphs ────────────────────────────────
# Streamlit / run(): in-process MemorySaver keeps per-thread state (e.g. the
# current problem) between turns.
dsa_coach_graph = build_graph(use_checkpointer=True)

# `langgraph dev` / Studio manage persistence themselves and reject graphs that
# carry their own checkpointer, so expose a checkpointer-free build for them.
graph = build_graph(use_checkpointer=False)


def run(
    question: str,
    *,
    mode: str = "",
    language: str = "python",
    student_code: str = "",
    topic: str | None = None,
    difficulty: str | None = None,
    session_id: str = "default",
    user_id: str = "student",
) -> dict:
    """
    Public entry point.  Returns the final state dict.

    Example:
        from agents.graph import run
        result = run("Explain BFS vs DFS", mode="learn")
        print(result["final_response"])
    """
    initial_state: AgentState = {
        "question":    question,
        "raw_question": question,
        "mode":        mode,
        "language":    language,
        "student_code": student_code,
        "topic":       topic,
        "difficulty":  difficulty,
        "session_id":  session_id,
        "user_id":     user_id,
        "thread_id":   session_id,
        "iteration":   0,
        "loop_count":  0,
        "retry_count": 0,
        # per-turn keys: a checkpointed thread would otherwise leak last turn's values
        "draft_response": "",
        "critique":    "",
        "needs_retry": False,
        "scorecard":   {},
    }
    # The checkpointer requires a thread_id — without it invoke() raises ValueError.
    config = {"configurable": {"thread_id": session_id}}
    return dsa_coach_graph.invoke(initial_state, config=config)
