"""
ui/app.py — DSA Coach Streamlit UI

Merges UI patterns from:
  - Root CodeCurry  : code editor, scorecard display, score history chart
  - AURA++          : session-aware chat with conversation history
  - lumina          : sidebar mode selector, topic/difficulty filters
  - AgentOps        : multi-mode tabs

Run:  streamlit run ui/app.py
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from agents.graph import run as _run_graph
from llm import optim
from llm.usage import tracker
from memory.store import load_progress, clear_conversation


def run(**kwargs):
    """Run the graph and remember how many tokens that turn cost."""
    mark = tracker.mark()
    result = _run_graph(**kwargs)
    st.session_state.last_usage = tracker.totals_for(tracker.records[mark:])
    return result


def show_usage() -> None:
    u = st.session_state.get("last_usage")
    if not u:
        return
    msg = (f"🔢 {u['calls']} LLM call(s) · {u['total_tokens']:,} tokens "
           f"({u['prompt_tokens']:,} in / {u['completion_tokens']:,} out)")
    if u["cache_hits"]:
        msg += f" · ⚡ {u['cache_hits']} cache hit(s), ~{u['saved_by_cache']:,} tokens saved"
    st.caption(msg)

# ----------------------------------------------------------------
# Page config
# ----------------------------------------------------------------
st.set_page_config(
    page_title="DSA Coach",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (role, content)

if "current_problem" not in st.session_state:
    st.session_state.current_problem = ""

if "student_code" not in st.session_state:
    st.session_state.student_code = ""

# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
with st.sidebar:
    st.title("🧠 DSA Coach")
    st.markdown("*Your intelligent coding mentor*")
    st.divider()

    mode = st.selectbox(
        "Mode",
        options=["", "learn", "practice", "hint", "solution", "code_review"],
        format_func=lambda x: {
            "":            "🤖 Auto-detect",
            "learn":       "📖 Learn a Concept",
            "practice":    "💪 Practice Problem",
            "hint":        "💡 Get a Hint",
            "solution":    "✅ View Solution",
            "code_review": "🔍 Code Review",
        }.get(x, x),
    )

    topic = st.selectbox(
        "Topic",
        ["", "arrays", "strings", "linked_list", "trees", "graphs",
         "dynamic_programming", "binary_search", "greedy", "backtracking",
         "stacks", "queues", "heaps", "tries"],
        format_func=lambda x: x.replace("_", " ").title() if x else "Any",
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["", "easy", "medium", "hard"],
        format_func=lambda x: x.title() if x else "Any",
    )

    language = st.selectbox(
        "Language",
        ["python", "java", "cpp", "javascript", "go"],
    )

    st.divider()
    optimized = st.toggle("⚡ Token optimizations", value=True,
                          help="Flip this and ask the same thing again to compare token usage. "
                               "Flags are process-wide (fine for a local demo).")
    optim.set_profile("optimized" if optimized else "baseline")
    totals = tracker.totals()
    st.metric("Tokens used (this server run)", f"{totals['total_tokens']:,}",
              help=f"{totals['calls']} LLM calls, {totals['cache_hits']} cache hits")

    st.divider()
    progress = load_progress()
    st.metric("Total Submissions", progress.get("submissions", 0))
    st.metric("Average Score", f"{progress.get('avg_score', 0):.1f} / 10")

    hist = progress.get("score_history", [])
    if hist:
        st.line_chart(
            {"Score": hist[-20:]},
            height=120,
            use_container_width=True,
        )

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        clear_conversation(st.session_state.session_id)
        st.session_state.chat_history = []
        st.session_state.current_problem = ""
        st.session_state.student_code = ""
        st.rerun()

# ----------------------------------------------------------------
# Main area — tabs
# ----------------------------------------------------------------
tab_chat, tab_code, tab_problem = st.tabs(["💬 Chat", "💻 Code Editor", "📋 Problem"])

# ── Chat tab ─────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask your DSA Coach")

    # Render conversation history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    # Input
    user_input = st.chat_input("Ask a question, request a problem, or paste your code…")

    if user_input:
        # Show user message immediately
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run graph
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = run(
                    question=user_input,
                    mode=mode,
                    language=language,
                    student_code=st.session_state.student_code,
                    topic=topic or None,
                    difficulty=difficulty or None,
                    session_id=st.session_state.session_id,
                )

            response = result.get("final_response", "") or result.get("answer", "")
            st.markdown(response)
            show_usage()

            # Show scorecard if code was reviewed
            scorecard = result.get("scorecard")
            if scorecard:
                cols = st.columns(5)
                labels = ["Correctness", "Efficiency", "Readability", "Approach", "Total"]
                maxes  = [4, 3, 2, 1, 10]
                for col, label, mx in zip(cols, labels, maxes):
                    val = scorecard.get(label, 0)
                    col.metric(label, f"{val}/{mx}")

            # Persist new problem context
            if result.get("problem"):
                st.session_state.current_problem = result["problem"]

        st.session_state.chat_history.append(("assistant", response))

# ── Code Editor tab ───────────────────────────────────────────────
with tab_code:
    st.subheader("Code Editor")
    st.caption("Write or paste your solution here, then switch to Chat and ask for a Code Review.")

    lang_mode = {"python": "python", "java": "java", "cpp": "c++",
                 "javascript": "javascript", "go": "go"}.get(language, "python")

    code_input = st.text_area(
        "Your code",
        value=st.session_state.student_code,
        height=400,
        placeholder=f"# Write your {language} solution here…",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 Save Code"):
            st.session_state.student_code = code_input
            st.success("Code saved! Now ask for a Code Review in the Chat tab.")
    with col2:
        if st.button("🚀 Submit for Review"):
            st.session_state.student_code = code_input
            with st.spinner("Reviewing your code…"):
                result = run(
                    question="Please review my code",
                    mode="code_review",
                    language=language,
                    student_code=code_input,
                    session_id=st.session_state.session_id,
                )
            response = result.get("final_response", "")
            st.markdown(response)
            show_usage()
            scorecard = result.get("scorecard")
            if scorecard:
                cols = st.columns(5)
                labels = ["Correctness", "Efficiency", "Readability", "Approach", "Total"]
                maxes  = [4, 3, 2, 1, 10]
                for col, label, mx in zip(cols, labels, maxes):
                    col.metric(label, f"{scorecard.get(label, 0)}/{mx}")

# ── Problem tab ───────────────────────────────────────────────────
with tab_problem:
    st.subheader("Current Problem")
    if st.session_state.current_problem:
        st.markdown(st.session_state.current_problem)
    else:
        st.info("Request a problem in the Chat tab (e.g. 'Give me a medium DP problem').")

    if st.button("🔄 Get New Problem"):
        with st.spinner("Fetching problem…"):
            result = run(
                question=f"Give me a {difficulty or 'medium'} {topic or ''} problem",
                mode="practice",
                topic=topic or None,
                difficulty=difficulty or None,
                session_id=st.session_state.session_id,
            )
        response = result.get("final_response", "")
        st.session_state.current_problem = result.get("problem", response)
        st.session_state.chat_history.append(("assistant", response))
        st.rerun()
