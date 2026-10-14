---
name: langgraph-local-deploy
description: >
  Step-by-step guide for building and deploying a LangGraph agent project on a local machine —
  from project scaffolding, dependency installation, and graph compilation through running the
  LangGraph dev server, LangGraph Studio, and a FastAPI/Streamlit frontend locally.
  Use this skill whenever the user wants to: run LangGraph locally, set up a LangGraph project
  from scratch, start the LangGraph dev server, connect LangGraph Studio to a local agent,
  deploy a LangGraph app on their own machine, debug a local LangGraph graph, configure
  environment variables for LangGraph, or wire a Streamlit/FastAPI UI to a local LangGraph graph.
  Trigger even when the user just says "run my agent locally", "how do I start LangGraph",
  "set up LangGraph on my laptop", or similar — this skill applies broadly to any local
  LangGraph development and deployment workflow.
---

# LangGraph — Local Build & Deploy

This skill covers the full local lifecycle:

1. [Project scaffold & dependencies](#1-project-scaffold--dependencies)
2. [Graph definition checklist](#2-graph-definition-checklist)
3. [Environment variables](#3-environment-variables)
4. [Run modes (choose one)](#4-run-modes)
   - 4a. Python script / REPL
   - 4b. LangGraph dev server (`langgraph dev`)
   - 4c. LangGraph Studio (desktop GUI)
   - 4d. FastAPI wrapper
   - 4e. Streamlit UI
5. [Checkpointer & persistence](#5-checkpointer--persistence)
6. [Common errors & fixes](#6-common-errors--fixes)

For Docker-based local deployment see → `references/docker.md`
For LangGraph Cloud / remote deployment see → `references/cloud.md`

---

## 1. Project scaffold & dependencies

### Recommended layout

```
my_agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py        ← compiled graph lives here
│   ├── nodes.py
│   ├── state.py
│   └── router.py
├── memory/
│   └── store.py
├── rag/
│   └── retriever.py
├── ui/
│   └── app.py          ← Streamlit or FastAPI
├── data/               ← auto-created at runtime
├── .env
├── pyproject.toml      ← project config + LangGraph settings
└── README.md
```

### Install

```bash
# Create and activate a virtual environment (strongly recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install from pyproject.toml (see below for the full file)
pip install -e ".[dev]"
```

### pyproject.toml — single source of truth

Replace `langgraph.json` + `requirements.txt` with a single `pyproject.toml`:

```toml
[project]
name = "my-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # Core LangGraph
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "langgraph-cli>=0.1.0",

    # LLM backends — keep both, switch via LLM_BACKEND env var
    "ollama>=0.3.0",
    "google-genai>=0.3.0",

    # Persistence
    "langgraph-checkpoint-sqlite",   # SQLite — zero setup, great for local
    # "langgraph-checkpoint-postgres", # uncomment for PostgreSQL

    # Utilities
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
rag = [
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "psycopg[binary]>=3.1",          # for pgvector
]
ui = [
    "streamlit>=1.35.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
]
dev = [
    "my-agent[rag,ui]",              # pulls in all extras
    "pytest>=8.0",
    "httpx>=0.27",                   # for FastAPI test client
]

# ── LangGraph configuration ──────────────────────────────────────
# This replaces langgraph.json entirely.
# `langgraph dev` and LangGraph Studio read [tool.langgraph] automatically.

[tool.langgraph]
# Each key is the graph name shown in Studio / used in API calls
# Value format: "path/to/module.py:variable_name"
[tool.langgraph.graphs]
agent = "./agent/graph.py:graph"
# Add more graphs as needed:
# reviewer = "./agent/reviewer_graph.py:reviewer_graph"

# Path to your .env file
env = ".env"

# Optional: pin the Python version for langgraph dev's sandbox
# python_version = "3.11"

# ── Build system ─────────────────────────────────────────────────
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[tool.setuptools.packages.find]
where = ["."]
include = ["agent*", "memory*", "rag*", "ui*"]
```

**After editing `pyproject.toml`**, re-install so the package index updates:
```bash
pip install -e ".[dev]"
```

---

## 2. Graph definition checklist

Every LangGraph project needs these three things before it can run:

### 2a. State (`agent/state.py`)
```python
from typing import TypedDict, List, Optional

class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    intent: str
    context: str
    draft_response: str
    final_response: str
    # … add your fields
```

### 2b. Nodes (`agent/nodes.py`)
Each node is a plain Python function: `(state: AgentState) -> AgentState`

```python
def my_node(state: AgentState) -> AgentState:
    # do work
    return {**state, "draft_response": "result"}
```

### 2c. Compiled graph (`agent/graph.py`)

```python
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import my_node, final_node

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("my_node", my_node)
    g.add_node("final",   final_node)
    g.add_edge(START,      "my_node")
    g.add_edge("my_node",  "final")
    g.add_edge("final",    END)
    return g.compile()

graph = build_graph()          # module-level — required for langgraph dev + Studio
```

**The module-level `graph` variable is required** when using `langgraph dev` or Studio.

---

## 3. Environment variables

Create a `.env` file in the project root (never commit this):

```dotenv
# LLM backend
LLM_BACKEND=ollama              # or "gemini"
OLLAMA_MODEL=llama3.2
GEMINI_API_KEY=your_key_here

# RAG backend
RAG_BACKEND=memory              # or "chroma" / "pgvector"
DATABASE_URL=postgresql://localhost/my_agent

# LangSmith tracing (optional but highly recommended for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=my-agent-local

# LangGraph dev server
LANGGRAPH_HOST=127.0.0.1
LANGGRAPH_PORT=8123
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()   # call once at app entry point
```

Install `python-dotenv` if not already present: `pip install python-dotenv`

---

## 4. Run modes

### 4a. Python script / REPL (simplest)

```python
# run.py
from dotenv import load_dotenv
load_dotenv()

from agent.graph import graph

result = graph.invoke({
    "question": "Explain binary search",
    "session_id": "test-001",
})
print(result["final_response"])
```

```bash
python run.py
```

### 4b. LangGraph dev server (`langgraph dev`)

The dev server exposes your graph over HTTP and enables hot-reload.

**Step 1 — confirm `[tool.langgraph]` is in `pyproject.toml`** (see Section 1):

```toml
[tool.langgraph.graphs]
agent = "./agent/graph.py:graph"
env = ".env"
```

- `agent` = graph name shown in Studio and used in API calls
- `"./agent/graph.py:graph"` = module path, colon, compiled graph variable name
- No separate `langgraph.json` needed — `langgraph dev` reads `pyproject.toml` directly

**Step 2 — start the server:**

```bash
langgraph dev
```

Or with explicit options:
```bash
langgraph dev --port 8123 --host 127.0.0.1 --reload
```

Server starts at `http://127.0.0.1:8123`. You'll see:
```
✓ LangGraph API server started
✓ Graph loaded: agent
  API: http://127.0.0.1:8123
  Docs: http://127.0.0.1:8123/docs
```

**Call via HTTP:**
```bash
curl -X POST http://127.0.0.1:8123/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"assistant_id":"agent","input":{"question":"Hello"}}'
```

Or via the Python SDK:
```python
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:8123")
async for chunk in client.runs.stream(
    None, "agent",
    input={"question": "Hello"},
    stream_mode="values",
):
    print(chunk.data)
```

### 4c. LangGraph Studio (desktop GUI)

Studio gives you a visual graph explorer, live state inspector, and time-travel debugging.

**Prerequisites:**
- `langgraph dev` must be running (Studio connects to it)
- macOS: download [LangGraph Studio desktop app](https://studio.langchain.com)
- Windows/Linux: use the web Studio at `https://smith.langchain.com/studio` pointing at your local server URL

**Connect Studio:**
1. Open LangGraph Studio
2. Click "Open existing project"
3. Select your project folder (the one containing `pyproject.toml`)
4. Studio auto-discovers the dev server and loads your graph

**What you can do in Studio:**
- Visualise the graph as a flowchart
- Send inputs and watch state flow node-by-node
- Inspect state at each checkpoint
- Time-travel: replay from any past checkpoint
- Edit state mid-run and re-run from that point

### 4d. FastAPI wrapper

Expose your graph as a REST API for integration with other services:

```python
# ui/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from agent.graph import graph

app = FastAPI(title="My Agent API")

class Request(BaseModel):
    question: str
    session_id: str = "default"
    language: str = "python"

class Response(BaseModel):
    answer: str
    session_id: str

@app.post("/ask", response_model=Response)
def ask(req: Request):
    result = graph.invoke({
        "question":   req.question,
        "session_id": req.session_id,
        "language":   req.language,
    })
    return Response(
        answer=result.get("final_response", ""),
        session_id=req.session_id,
    )

@app.get("/health")
def health():
    return {"status": "ok"}
```

```bash
uvicorn ui.api:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`

### 4e. Streamlit UI

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`

Key Streamlit pattern for LangGraph:
```python
import streamlit as st
from agent.graph import graph

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask your agent…")
if user_input:
    with st.spinner("Thinking…"):
        result = graph.invoke({
            "question":   user_input,
            "session_id": st.session_state.get("session_id", "default"),
        })
    response = result.get("final_response", "")
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("assistant", response))

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)
```

---

## 5. Checkpointer & persistence

Without a checkpointer, every `graph.invoke()` starts fresh. Add one to enable:
- Multi-turn conversations
- Time-travel debugging in Studio
- State persistence across restarts

### SQLite (recommended for local dev — zero setup)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("data/checkpoints.db") as checkpointer:
    graph = build_graph().compile(checkpointer=checkpointer)
```

Or for long-running servers, use a context-manager-free form:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = build_graph().compile(checkpointer=checkpointer)
```

### In-memory (dev/testing only — resets on restart)

```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
graph = build_graph().compile(checkpointer=checkpointer)
```

### PostgreSQL (production-grade local)

```bash
pip install langgraph-checkpoint-postgres psycopg[binary]
```

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

conn = Connection.connect("postgresql://user:pass@localhost/mydb", autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()   # creates tables on first run
graph = build_graph().compile(checkpointer=checkpointer)
```

### Using thread_id for multi-turn sessions

Pass a `config` dict with `thread_id` on each invoke:

```python
config = {"configurable": {"thread_id": "user-abc-session-1"}}
result = graph.invoke({"question": "Hello"}, config=config)

# Next turn — graph loads previous state automatically
result2 = graph.invoke({"question": "Follow up"}, config=config)
```

---

## 6. Common errors & fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langgraph` | Not installed in active venv | `pip install langgraph` in the correct venv |
| `No module named 'agent'` | Running from wrong directory | `cd` to project root before running |
| `langgraph: command not found` | CLI not installed | `pip install langgraph-cli` |
| `Graph not found: agent` | `[tool.langgraph.graphs]` path wrong | Check module path and variable name in `pyproject.toml` |
| `Port 8123 already in use` | Another process has the port | `lsof -i :8123` then kill, or `--port 8124` |
| Checkpointer `thread_id` missing | No config passed to invoke | Add `config={"configurable":{"thread_id":"…"}}` |
| Studio shows empty graph | No module-level `graph` variable | Ensure `graph = build_graph()` at module level |
| `LANGCHAIN_API_KEY not set` | Tracing enabled but no key | Either set the key or `LANGCHAIN_TRACING_V2=false` |
| SQLite `check_same_thread` error | Using SQLite in FastAPI threads | Pass `check_same_thread=False` to `sqlite3.connect()` |
| Ollama `connection refused` | Ollama not running | `ollama serve` in a separate terminal |

---

## Quick-start checklist

```
□ python -m venv .venv && source .venv/bin/activate
□ Create pyproject.toml with [project] deps + [tool.langgraph] config
□ pip install -e ".[dev]"
□ Create agent/state.py, agent/nodes.py, agent/graph.py
□ Add module-level `graph = build_graph()` to graph.py
□ Create .env with LLM keys
□ langgraph dev --reload
□ Open LangGraph Studio → select project folder
□ Test with curl or the SDK
□ Add SQLite checkpointer for persistence
```