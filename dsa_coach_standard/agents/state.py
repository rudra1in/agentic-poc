"""
state.py — Unified AgentState for DSA Coach

Merges the best fields from:
  - Root CodeCurry   (scorecard, score_history, progress)
  - AURA++           (session_id, conversation_history, ReAct fields)
  - lumina           (intent, critic/retry loop, draft_response)
  - AgentOps         (agent_type, retrieved_documents, evaluation, thread_id)
"""

from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator


class AgentState(TypedDict, total=False):

    # ----------------------------------------------------------------
    # User input
    # ----------------------------------------------------------------
    question: str          # working question (a critic retry appends the critique here)
    raw_question: str      # what the student actually typed (saved to history)
    mode: str              # sidebar mode: learn | practice | hint | solution | code_review
    language: str          # programming language (python, java, cpp …)
    student_code: str      # code submitted by the student

    # ----------------------------------------------------------------
    # Session / identity
    # ----------------------------------------------------------------
    session_id: str        # maps to a student session (AURA++)
    user_id: str           # unique student identifier
    thread_id: str         # LangGraph checkpointer thread (AgentOps)

    # ----------------------------------------------------------------
    # Problem context
    # ----------------------------------------------------------------
    problem_id: str
    problem: str           # full problem description text
    topic: Optional[str]   # arrays | trees | dp | graphs …
    difficulty: Optional[str]  # easy | medium | hard

    # ----------------------------------------------------------------
    # Intent routing
    # ----------------------------------------------------------------
    intent: str            # learn | practice | hint | solution | code_review | direct
    agent_type: str        # coach | code | hint | mcq | roadmap | interview

    # ----------------------------------------------------------------
    # ReAct / multi-step planning (AURA++)
    # ----------------------------------------------------------------
    route: str             # initial route decision
    next_action: str       # planner's chosen next action
    observation: str       # latest tool observation
    tool_result: str       # raw tool output
    iteration: int         # current ReAct loop iteration
    max_iterations: int    # safety cap (default 5)

    # ----------------------------------------------------------------
    # RAG
    # ----------------------------------------------------------------
    retrieved_knowledge: List[str]          # text chunks (root / AURA++)
    retrieved_documents: List[Any]          # full doc objects (AgentOps)
    context: str                            # assembled RAG context string

    # ----------------------------------------------------------------
    # Conversation memory
    # ----------------------------------------------------------------
    conversation_history: List[Dict[str, Any]]   # AURA++ per-session chat
    conversation: Annotated[                      # AgentOps appended list
        List[Dict[str, Any]],
        operator.add,
    ]
    history: List[str]                           # lumina lightweight history

    # ----------------------------------------------------------------
    # Agent processing — draft & critic loop (lumina / AgentOps)
    # ----------------------------------------------------------------
    plan: str
    draft_response: str
    critique: str
    needs_retry: bool
    loop_count: int

    # ----------------------------------------------------------------
    # Evaluation / self-correction (AgentOps)
    # ----------------------------------------------------------------
    evaluation: str        # "good" | "bad"
    retry_count: int
    max_retries: int
    evaluation_override: str   # testing only

    # ----------------------------------------------------------------
    # Scorecard (root CodeCurry)
    # ----------------------------------------------------------------
    scorecard: Dict[str, int]   # {Correctness, Efficiency, Readability, Approach, Total}
    score: int
    score_history: List[int]
    avg_score: float

    # ----------------------------------------------------------------
    # Final output
    # ----------------------------------------------------------------
    answer: str
    intro: str
    analysis: str
    complexity: str
    hint: str
    feedback: str
    encouragement: str
    final_response: str
    done: bool
