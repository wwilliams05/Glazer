# Glazer: a Luka Dončić stat agent

Predicts Luka Dončić's points, rebounds and assists for his next game, explains the prediction, and paper-trades it against Kalshi and Polymarket player-prop markets to see whether it beats them. **No real money:** the market APIs are only ever read.

The full spec is in [docs/PRD.md](docs/PRD.md) and the build plan in [docs/PLAN.md](docs/PLAN.md).

## Status

Week 1 of 6 is done (November 2026).

- **Database:** MongoDB Atlas, with validators and indexes defined in code ([luka/db/schema.py](luka/db/schema.py)).
- **Market prices:** Kalshi and Polymarket prices for Luka's props are saved every 15 minutes before each game. Pre-game prices can't be recovered later, so collection started first.
- **Game logs:** all 581 of Luka's career games, from `stats.nba.com`, updated daily.
- **Baseline:** guessing his season average to date is off by **6.6 points, 2.5 rebounds and 2.6 assists** per game on average (mean absolute error). Every model has to beat that.

Next: features and the first models (week 2), then a chat agent, scoring, the paper-trading engine and a research agent. The agent design is in [docs/AGENTS.md](docs/AGENTS.md).

## How it fits together

```
Kalshi, Polymarket ──► market snapshots (every 15 min, GitHub Actions + local backup)
stats.nba.com ───────► game logs (daily, local)          │
                                                          ▼
                        MongoDB Atlas  ◄── features ◄── model ◄── rules engine ──► paper trades
                              ▲
                     chat agent (read-only tools)
```

Ingestion, features, predictions, bet decisions and scoring are plain Python. An LLM (Claude) is used only where judgment is needed: reading injury news, writing trade explanations and answering questions. It never writes to the database or decides a bet.

The rules that keep the results honest:

- **No lookahead.** Every feature, prediction and trade uses only data timestamped before the decision. The database's validators reject a trade priced after it was decided or decided after tip-off.
- **Time-ordered validation.** Models are tested on games that come after the ones they were trained on.
- **Frozen strategies.** A change to a betting rule creates a new strategy version, so results are never mixed.

## Running it

Needs Python 3.11 and a MongoDB Atlas cluster (the free M0 tier is enough).

```
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
copy .env.example .env          # then set MONGODB_URI
.venv/Scripts/python -m luka.db.client     # check the connection
.venv/Scripts/python -m luka.db.schema     # create collections, validators, indexes
```

| Command | What it does |
|---|---|
| `python -m luka.ingest.games` | Load the full career game log (`--current` for this season only) |
| `python -m luka.ingest.markets` | Save one snapshot of Luka's open prop markets (`--dry-run` to print only) |
| `python -m luka.model.baseline` | Print the season-average baseline error |
| `python -m pytest -q` | Run the tests (database tests run only when `MONGODB_URI` is set) |

### Scheduled jobs

| Job | Where | Why there |
|---|---|---|
| Market snapshots, every 15 min | [GitHub Actions](.github/workflows/market-snapshots.yml), plus a Windows scheduled task as backup | GitHub delays and skips frequent schedules, so the laptop fills the gaps |
| Game logs, daily | Windows scheduled task ([scripts/run_logged.py](scripts/run_logged.py) running `luka.ingest.games --current`) | `stats.nba.com` blocks GitHub's servers |

The GitHub job needs a repository secret `MONGODB_URI`, and Atlas Network Access must allow `0.0.0.0/0`.

## Layout

```
luka/
  db/        connection and schema (collections, validators, indexes)
  ingest/    games.py (stats.nba.com), markets.py (Kalshi, Polymarket)
  model/     baseline.py
scripts/     local scheduled-task entry points
tests/
docs/        PRD, build plan, agent design, market API notes
.claude/     Claude Code helper subagents used while building
```

## Built with

Python, MongoDB Atlas, `nba_api`, pandas, scikit-learn (from week 2), the Claude API (from week 3), and Claude Code.
