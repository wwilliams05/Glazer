# Luka Stat Agent

Predicts Luka Dončić's PTS, REB and AST, explains why, and paper-trades the predictions against Kalshi and Polymarket. Spec: [docs/PRD.md](docs/PRD.md). Current milestone: [docs/PLAN.md](docs/PLAN.md).

## Hard rules

- **No real money.** Market APIs are read-only. Never call Kalshi or Polymarket order, portfolio or auth endpoints.
- **Secrets live in `.env` only** (see `.env.example`). Never hardcode, print or commit keys or connection strings.
- **No lookahead.** Any feature, prediction or trade decision for a game uses only data timestamped before its `as_of` / decision time. Never use same-game minutes as a feature. Ignore market strikes that opened after tip-off.
- **Time-ordered validation only.** Never shuffle games across a train/test split.
- **Schema is the contract.** Validators and indexes live in `db/schema.py`. Change them there; never bypass validation.
- **Strategy configs are immutable.** A rule change means a new version doc.
- **LLMs only where judgment is needed.** Ingestion, features, prediction, bet decisions and scoring are deterministic Python.
- **Chat answers:** every number comes from a tool result; always show the range; never call a pick a lock.

## Commands

- Setup: `py -3.11 -m venv .venv`, then `.venv/Scripts/python -m pip install -r requirements.txt`
- Tests: `.venv/Scripts/python -m pytest -q` (Atlas tests skip unless `MONGODB_URI` is set; they use the `luka_test` database)
- Apply schema: `.venv/Scripts/python -m luka.db.schema` (`--dry-run` to preview)
- Rebuild features: `.venv/Scripts/python -m luka.features.pipeline`; train and evaluate: `.venv/Scripts/python -m luka.model.train`
- Reload the schedule now: `.venv/Scripts/python -m luka.ingest.schedule`
- Prefix Python runs with `PYTHONIOENCODING=utf-8`, or printing "Dončić" crashes the Windows console.

## Operations

- **Market snapshots** run every 15 min two ways: GitHub Actions (`.github/workflows/market-snapshots.yml`, unreliable timing) and the Windows task "Glazer market snapshots" (`pythonw scripts/run_logged.py markets_snapshots.log luka.ingest.markets`). The same run then calls `luka.model.predict`, which writes live predictions ~2 h before tip-off; on days with no game within 30 h it exits after one query.
- **Game logs, team data and schedule** run daily via the Windows task "Glazer game logs" (`pythonw scripts/run_logged.py games_update.log luka.ingest.games --current`). stats.nba.com blocks GitHub's servers, so this can't move to Actions.
- **Predictions are write-once.** Never update or delete a `predictions` doc before its game. Check a prediction without writing: `python -m luka.model.predict --dry-run --game-id <id>`. `luka/model/cards/m2.json` is frozen; a model change is a new `MODEL_VERSION`.
- Logs (`*.log`) are in the repo root and git-ignored. `scripts/run_logged.py` waits for the network before running.
- Claude Code's permission check blocks creating or changing Windows scheduled tasks. Give the owner a PowerShell command to paste instead.
- **Git:** one branch per milestone (e.g. `week2-features`), commit per step with tests passing, then fast-forward `main` and push.
- **Evaluation:** games from 2025-26 to 2026-11-22 are no longer a clean holdout (see PLAN.md, week 2). Live games from 2026-11-23 on are.

## Conventions

- Python 3.11, package code under `luka/`.
- Writes are idempotent: upsert on natural keys (`game_id`, `game_id + stat + model_version`).
- Store all timestamps in UTC.
- Cache raw API responses in `raw_api_cache` (TTL) before normalizing.

## External APIs (details in [docs/market-api-spike.md](docs/market-api-spike.md))

- Kalshi: `https://api.elections.kalshi.com/trade-api/v2`. Series `KXNBAPTS`, `KXNBAREB`, `KXNBAAST` are threshold ladders (`floor_strike` = k − 0.5). Archived markets live under `/historical/...` and use different field names than the live API.
- Polymarket: Gamma API for discovery (`/events?slug=nba-xxx-yyy-YYYY-MM-DD`), CLOB `/prices-history` for prices. Closed markets need explicit `startTs` and `endTs`.
- `nba_api` (stats.nba.com) is unofficial and often blocks cloud IPs. Add retries and a timeout.

## Working style

- One milestone at a time; tests pass before commit.
- The owner is learning. After writing non-obvious code, briefly explain what it does and why.
