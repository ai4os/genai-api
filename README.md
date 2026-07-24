# genai-api

[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitHub license](https://img.shields.io/github/license/MartaOB/genai-api.svg)](https://github.com/MartaOB/genai-api/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/MartaOB/genai-api.svg)](https://github.com/MartaOB/genai-api/releases)


> ⚠️ **Warning**
>
> This repository is under active development. APIs, configuration, and internal
> implementation details may change without notice.  Some features may be
> incomplete or not fully functional.

API that provides a stable backend for chat management, AI agents, MCP tools, databases and other AI services.

## Architecture

`genai-api` is one piece of a larger system:

- **Frontend** ([`genai-ui`](https://github.com/ai4os/genai-ui)) — chat interface that consumes this API.
- **Backend** (this repo) — FastAPI backend: chats, messages, model selection.
- **Keycloak** — authentication provider. Endpoints are protected with JWTs validated against Keycloak's JWKS.
- **LiteLLM** — gateway to the actual LLM providers. `genai-api` calls it via `httpx`.
- **PAPI** ([`ai4-papi`](https://github.com/ai4os/ai4-papi)) — internal API that manages per-user LiteLLM API keys. `genai-api` calls it on the user's behalf (using their token) to fetch or create their LiteLLM key, and caches it locally.

## Requirements

- Python 3.11+
- Docker / Docker Compose (for local PostgreSQL)
- Access to a Keycloak realm with a client configured
- Access to a LiteLLM instance and to PAPI (`ai4papi`)

## Installation

Install this package in editable mode running:

```bash
pip install -e .
```

## Configuration

The app reads its configuration from environment variables stored in a `.env file` (see `genai_api/core/config.py`).

```bash
DATABASE_URL=postgresql+asyncpg://genai:genai@localhost:5432/genai_dev
KEYCLOAK_ISSUER=https://<your-keycloak-host>/realms/<realm>
LITELLM_BASE_URL=https://<your-litellm-host>
LITELLM_API_KEY=<litellm-service-key>
PAPI_BASE_URL=https://<your-papi-host>
```

For running the test suite, an equivalent `.env.test` file is expected, pointing at the `genai_test` database instead of `genai_dev`.

## Local deployment

1. **Create and activate a virtual environment**

```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
```

2. **Start PostgreSQL** (spins up both the `genai_dev` and `genai_test` databases)

```bash
   docker compose up -d
```

3. **Set up your environment variables**

   Copy `.env.example` to `.env` and fill in the values described in [Configuration](#configuration).

4. **Apply database migrations**

```bash
   alembic upgrade head
```

5. **Run the API**

```bash
   uvicorn genai_api.main:app --reload
```

   The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs` and a health check at `http://localhost:8000/health`.

### Running the tests

Tests run against the separate `genai_test` database and don't require a real Keycloak/PAPI/LiteLLM connection (external calls are mocked).

```bash
set -a; source .env.test; set +a
pytest
```


## How to contribute

The repository is formatted with [Ruff](https://docs.astral.sh/ruff/).
We use [Pre-commit](https://pre-commit.com/) to enforce correct formatting in new contributions.
To automatically run locally the pre-commit checks before committing, install the custom pre-commit workflow:

```bash
pip install -r requirements-dev.txt
pre-commit install
```
