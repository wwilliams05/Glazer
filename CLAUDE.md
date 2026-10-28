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
