# LangGraph Cloud & Remote Deployment

For local deployment use the main SKILL.md. This reference covers moving beyond local.

## LangGraph Cloud (managed)

LangGraph Cloud is LangChain's hosted platform for LangGraph agents.

### Deploy to LangGraph Cloud

```bash
# Install CLI (if not already)
pip install langgraph-cli

# Login
langgraph login

# Deploy (from project root — reads [tool.langgraph] from pyproject.toml)
langgraph deploy
```

The CLI reads `[tool.langgraph]` from `pyproject.toml` and bundles your code automatically.

### Access your deployment

```python
from langgraph_sdk import get_client

client = get_client(url="https://your-deployment.langgraph.app")
result = await client.runs.stream(
    None, "agent",
    input={"question": "Hello"},
    stream_mode="values",
)
```

## Self-hosted on a VPS / server

Run the same `langgraph dev` command on a remote machine, then expose it:

```bash
# On the server
langgraph dev --host 0.0.0.0 --port 8123

# On your local machine — SSH tunnel for secure access
ssh -L 8123:localhost:8123 user@your-server
# Now http://localhost:8123 connects to the remote server
```

## Key differences: local vs cloud

| Feature | Local (`langgraph dev`) | LangGraph Cloud |
|---|---|---|
| Setup | `pip install langgraph-cli` | `langgraph login && deploy` |
| Persistence | SQLite / local Postgres | Managed Postgres |
| Scaling | Single process | Auto-scaling |
| Studio | Desktop app or web | Web Studio |
| Cost | Free | Pay-per-use |
| Hot reload | `--reload` flag | Redeploy |