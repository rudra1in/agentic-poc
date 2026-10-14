To address the ARB findings, here is the complete, production-grade package structure adopting the recommended **"Thick Brain, Micro-Tools"** architecture.

In this structure, the multi-agent reasoning loops and graph state transitions remain in-process for high speed and state consistency, while heavy, isolated dependencies (browser scrapers, sandbox runtimes, external APIs) are decoupled into containerized microservice tools.

---

### Complete Production Repository Structure

```text
multi_agent_platform/
├── .github/
│   └── workflows/
│       ├── ci-tests.yaml               # Lint, unit, and sensor test matrix
│       ├── cd-agent-engine.yaml        # Build & push core LangGraph image
│       └── cd-tool-services.yaml       # Build & push independent tool microservices
│
├── platform/                           # Infrastructure-as-Code & Deployments
│   ├── docker/
│   │   ├── Dockerfile.engine           # Core multi-agent LangGraph server
│   │   ├── Dockerfile.scraper          # Headless browser (Playwright) tool service
│   │   └── Dockerfile.sandbox          # Isolated code execution tool service
│   ├── k8s/
│   │   ├── base/
│   │   │   ├── configmaps.yaml
│   │   │   ├── secrets.yaml
│   │   │   └── ingress.yaml
│   │   ├── apps/
│   │   │   ├── agent-engine-deployment.yaml   # Scaled stateless agent pods
│   │   │   ├── scraper-tool-deployment.yaml   # Scalable browser farm
│   │   │   └── sandbox-tool-deployment.yaml   # Secure runner
│   │   ├── jobs/
│   │   │   └── checkpoint-pruner-cron.yaml    # Daily retention/TTL cleanup
│   │   └── backing-services/
│   │       ├── postgres-statefulset.yaml      # Checkpointer persistence
│   │       ├── pgbouncer-deployment.yaml      # Connection pooling proxy
│   │       ├── redis-deployment.yaml          # Distributed lock / rate limiter
│   │       └── otel-collector.yaml            # Tracing & metrics
│   └── docker-compose.yml              # Local complete stack emulator
│
├── shared_lib/                         # Cross-cutting Internal Python Package
│   ├── pyproject.toml
│   └── src/
│       └── platform_common/
│           ├── __init__.py
│           ├── schemas/
│           │   ├── __init__.py
│           │   ├── state.py            # TypedDict + add_messages reducer
│           │   └── errors.py           # Standardized error contracts
│           ├── telemetry/
│           │   ├── __init__.py
│           │   ├── tracer.py           # OpenTelemetry span initializers
│           │   └── metrics.py          # Prometheus counters (tokens, latency)
│           └── resilience/
│               ├── __init__.py
│               ├── retries.py          # Pre-configured RetryPolicy instances
│               └── circuit_breaker.py  # Sliding window failure breaker
│
├── services/                           # Microservices
│   │
│   ├── agent_engine/                   # Core Brain: Multi-Agent Orchestration Service
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                 # FastAPI server exposing runs/threads
│   │   │   ├── config.py               # Pydantic BaseSettings
│   │   │   │
│   │   │   ├── api/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── v1/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── threads.py      # POST /threads, GET /threads/{id}/state
│   │   │   │   │   ├── runs.py         # POST /threads/{id}/runs (streaming/SSE)
│   │   │   │   │   └── resume.py       # POST /threads/{id}/resume (HITL resume)
│   │   │   │   └── health.py           # /healthz (liveness), /readyz (DB ping)
│   │   │   │
│   │   │   ├── graph/                  # Graph Construction Layer
│   │   │   │   ├── __init__.py
│   │   │   │   ├── builder.py          # StateGraph assembly & compilation
│   │   │   │   ├── checkpoint.py       # AsyncPostgresSaver via pgBouncer pool
│   │   │   │   ├── nodes/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── supervisor.py   # Intent triage & routing node
│   │   │   │   │   ├── researcher.py   # Research sub-agent logic node
│   │   │   │   │   ├── analyst.py      # Analysis reasoning node
│   │   │   │   │   └── escalation.py   # Human-in-the-loop interrupt node
│   │   │   │   └── edges/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── routers.py      # Happy-path routing conditions
│   │   │   │       └── circuit.py      # Error-boundary conditional gates
│   │   │   │
│   │   │   └── tools/                  # Tool Client Wrappers (Calls Micro-Tools)
│   │   │       ├── __init__.py
│   │   │       ├── client_scraper.py   # Thin HTTP client to Scraper Service
│   │   │       ├── client_sandbox.py   # Thin HTTP client to Sandbox Service
│   │   │       └── client_datafabric.py# Client to Appian Data Fabric / ERP API
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   │
│   ├── scraper_service/                # Dedicated Micro-Tool #1 (Heavy Pod)
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                 # FastAPI scraper endpoints
│   │   │   └── scraper.py              # Playwright / Chromium headless logic
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   │
│   └── sandbox_service/                # Dedicated Micro-Tool #2 (Secure Pod)
│       ├── app/
│       │   ├── __init__.py
│       │   ├── main.py                 # Isolated execution endpoint
│       │   └── runner.py               # Sandboxed Python executor (e.g., E2B/gVisor)
│       ├── Dockerfile
│       └── pyproject.toml
│
├── tests/
│   ├── unit/
│   │   ├── test_reducers.py            # Validates state reducer & message integrity
│   │   └── test_circuit_breaker.py     # Verifies edge routing on consecutive failures
│   ├── integration/
│   │   ├── test_graph_flow.py          # End-to-end multi-agent execution mock
│   │   └── test_persistence.py        # Postgres checkpointer & time-travel recovery
│   └── sensors/                        # Architecture & Quality Sensors
│       ├── test_no_sync_http.py        # Fails if synchronous httpx is in agent nodes
│       └── test_arch_fitness.py        # Enforces separation of graph from API layers
│
├── langgraph.json                      # Local development via LangGraph Studio / CLI
├── pyproject.toml                      # Root workspace configuration
└── README.md

```

---

### Key File Specifications

#### 1. Correct State Schema (`shared_lib/.../state.py`)

Fixes the reducer flaw by using `TypedDict` and the built-in `add_messages` reducer to append messages without serialization loss:

```python
from typing import Annotated, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_step: Optional[str]
    tenant_id: str
    trace_id: str
    retry_count: int
    last_error: Optional[str]
    circuit_breaker_tripped: bool

```

#### 2. Connection-Pooled Checkpointer (`services/agent_engine/.../checkpoint.py`)

Uses async connection pooling to avoid exhausting Postgres limits when multiple threads execute concurrently:

```python
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.config import settings

pool = AsyncConnectionPool(
    conninfo=settings.DATABASE_URL,
    max_size=20,
    kwargs={"autocommit": True}
)

async def get_checkpointer() -> AsyncPostgresSaver:
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver

```

#### 3. Asynchronous Tool Microservice Client (`services/agent_engine/.../client_scraper.py`)

Node calls to tools reuse a persistent, managed `AsyncClient`:

```python
import httpx
from langchain_core.tools import tool
from app.config import settings

client = httpx.AsyncClient(base_url=settings.SCRAPER_SERVICE_URL, timeout=45.0)

@tool
async def scrape_url(url: str) -> str:
    """Fetches clean markdown from a URL via the headless scraper service."""
    response = await client.post("/scrape", json={"url": url})
    response.raise_for_status()
    return response.json()["markdown_content"]

```

#### 4. Clean Graph Builder with Resilience (`services/agent_engine/.../builder.py`)

```python
from langgraph.graph import StateGraph, START, END
from platform_common.schemas.state import MultiAgentState
from platform_common.resilience.retries import NETWORK_RETRY_POLICY
from app.graph.nodes.supervisor import supervisor_node
from app.graph.nodes.researcher import researcher_node
from app.graph.nodes.analyst import analyst_node
from app.graph.nodes.escalation import escalation_node
from app.graph.edges.routers import route_from_supervisor
from app.graph.edges.circuit import check_agent_resilience

def build_production_graph(checkpointer):
    workflow = StateGraph(MultiAgentState)

    # Nodes with retry policies for transient network drops
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node, retry=NETWORK_RETRY_POLICY)
    workflow.add_node("analyst", analyst_node, retry=NETWORK_RETRY_POLICY)
    workflow.add_node("escalation", escalation_node)

    # Core flow
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", route_from_supervisor)
    
    # Circuit-breaker returns
    workflow.add_conditional_edges("researcher", check_agent_resilience)
    workflow.add_conditional_edges("analyst", check_agent_resilience)

    return workflow.compile(checkpointer=checkpointer)

```