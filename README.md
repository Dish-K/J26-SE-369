# CodeTrace

CodeTrace is the J26-SE-369 research project for real-time behavior analysis of AI-assisted coding during live technical interviews. This repository currently provides the connected development foundation shared by the project team: a FastAPI backend, local PostgreSQL with TimescaleDB, a reusable asynchronous database pool, automated tests, and CI across the supported Python versions.

## Component responsibility areas

- **Telemetry Infrastructure:** Captures candidate coding behavior and produces behavioral data and features.
- **Behavioral Classifier:** Analyzes agreed behavioral inputs and produces classifier decisions and evidence.
- **Adaptive Probing:** Produces targeted post-submission follow-up probes from approved classifier outputs.
- **Explainability:** Presents explanations of classifier outputs to the interviewer.

These folders currently define ownership boundaries only. Component implementation is intentionally deferred.

## Prerequisites

Install:

- Git
- Docker with Docker Compose
- Python 3.11, 3.12, or 3.13

Python 3.14 is not currently supported by this project.

## Local setup

### 1. Clone the repository

```text
git clone https://github.com/Dish-K/J26-SE-369.git
cd J26-SE-369
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Python 3.12 or 3.13 may be used instead of Python 3.11.

Confirm that the virtual environment is active:

```text
python --version
```

### 3. Install Python dependencies

The repository uses one Python dependency file:

```text
python -m pip install --requirement backend/requirements.txt
```

Do not create or install a separate development requirements file.

### 4. Create the local environment file

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

The included values are for local development only. Review them before starting the services.

`.env` may contain local credentials and must never be committed. Only `.env.example` belongs in Git.

### 5. Start TimescaleDB

```text
docker compose up --detach --wait db
```

This starts only the local PostgreSQL 16–lineage TimescaleDB service and preserves its files in a named Docker volume.

Verify its status:

```text
docker compose ps db
```

The service should report `healthy`.

Verify PostgreSQL readiness directly:

```text
docker compose exec db sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

The result should end with:

```text
accepting connections
```

### 6. Start FastAPI

From the repository root, with the virtual environment active:

```text
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Startup is complete when Uvicorn reports:

```text
Application startup complete.
```

The backend opens the shared database pool during application startup and closes it during shutdown.

### 7. Verify the connected health endpoint

Windows PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health | ConvertTo-Json -Compress
```

macOS/Linux:

```bash
curl --fail --silent http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","database":"ok"}
```

This response confirms that FastAPI is running and can execute the initialization `SELECT 1` check through the shared `asyncpg` pool.

### 8. Run the tests

Keep the database service running, open another terminal, activate the same virtual environment, and run:

```text
python -m pytest -v
```

The connected health test uses FastAPI’s normal application lifecycle and expects TimescaleDB to be available.

To stop the local database without deleting its data:

```text
docker compose stop db
```

## Repository responsibilities

```text
.github/       GitHub Actions and repository automation
backend/       Live FastAPI backend application
contracts/     Future approved cross-component executable contracts
frontend/      Future candidate and interviewer frontend
ml/            Future offline data preparation, training, and evaluation work
tests/         Automated tests for implemented repository behavior
docs/          Engineering documentation and decision records
```

Root infrastructure files provide:

- `.env.example` — local development configuration template.
- `docker-compose.yml` — local TimescaleDB service only.
- `pyproject.toml` — supported Python range and pytest configuration.
- `backend/requirements.txt` — the single Python dependency file.

## Shared database foundation

`backend/app/db/pool.py` owns the reusable `asyncpg` connection pool used by the FastAPI application. It loads the agreed `DB_*` environment variables, creates and closes one shared pool through the FastAPI lifespan, and performs the lightweight `SELECT 1` connectivity check used by `/health`.

Component code should reuse this shared pool rather than open independent database connections.

## Intentionally deferred

Repository initialization does not provide:

- Database migrations, project tables, or an application schema.
- ORM models, repositories, or component-specific database queries.
- Executable shared contracts.
- Telemetry, classifier, probing, or explainability implementation.
- Data generators, training workflows, or model artifacts.
- A selected or scaffolded frontend framework.
- Deployment or cloud infrastructure.

These items require later component context and explicit team decisions.

## Contribution workflow

`main` is the only long-lived branch. Meaningful changes should use short-lived branches, be reviewed through pull requests, and pass CI before merging.

Codex acts as an advisory code generator during initialization. It does not directly create commits, push changes, open pull requests, create branches or tags, or change repository settings; a team member reviews and performs those actions.