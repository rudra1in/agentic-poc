# Docker — Local LangGraph Deployment

Use Docker when you want a reproducible, isolated environment or want to mimic a production deploy locally.

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps from pyproject.toml
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY . .

# Default: run langgraph dev server
EXPOSE 8123
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8123"]
```

## docker-compose.yml (with optional PostgreSQL)

```yaml
version: "3.9"
services:

  agent:
    build: .
    ports:
      - "8123:8123"
    env_file:
      - .env
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/agentdb
    volumes:
      - ./data:/app/data      # persist SQLite / data files
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: agentdb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 10

  ui:
    build: .
    command: streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    env_file:
      - .env
    depends_on:
      - agent

volumes:
  pgdata:
```

## Commands

```bash
# Build and start everything
docker compose up --build

# Start in background
docker compose up -d

# Tail logs
docker compose logs -f agent

# Stop
docker compose down

# Rebuild after code changes
docker compose up --build agent
```

## .dockerignore

```
.venv/
__pycache__/
*.pyc
.env
data/
*.db
.git/
*.egg-info/
dist/
```

## Connecting Studio to a Docker-based dev server

When `langgraph dev` runs inside Docker on port 8123 mapped to localhost:

1. Start with `docker compose up agent`
2. In LangGraph Studio, use URL: `http://localhost:8123`
3. Studio connects to the container's dev server exactly as it would to a local one