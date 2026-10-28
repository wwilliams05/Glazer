# Luka Stat Agent: PRD v0.3

**Owner:** Weston Williams · **Updated:** 2026-10-28 · Build plan: [PLAN.md](PLAN.md)

## Changes since v0.2

| Change | Why |
|---|---|
| `games` is a regular collection, not a time series | Time series collections can't have unique indexes, upserts or change streams, and about 500 docs gain nothing from bucketing |
| Market snapshot capture moves to week 1 | Point-in-time prices can't be backfilled |
| Kalshi is the primary benchmark; Polymarket is secondary | [Spike](market-api-spike.md): Kalshi has $10K–$145K volume per strike and a 51-game archive; Polymarket has $10–$1K |
| Model outputs a full distribution (empirical residuals) | Kalshi lists ladders (25+, 30+, …), so we need P(stat ≥ k) for every strike, plus consistent ranges |
| Teammate features are generic (`n_starters_out`, `usage_share_out`) | Luka's career covers two teams; named-player features don't transfer |
| GitHub Actions is the main scheduler; `nba_api` may run locally | Atlas Triggers run JavaScript only; `stats.nba.com` often blocks cloud IPs |
| One local `worker.py` listens to change streams | Change streams need a process that keeps running |

## One-liner

A personal AI agent that predicts Luka Dončić's points, rebounds and assists for his next game, explains why, and tracks its own accuracy against Kalshi and Polymarket lines. No real money involved.

## User and problem

**User:** me, a fan who checks prediction-market lines before Lakers games.
**Problem:** lines exist, but I have no transparent, data-backed view of my own, and no way to tell whether my intuition beats the market.

## Core flow (v1)

1. I ask: "What's Luka's line tonight vs. Portland?"
2. The agent pulls the next game, the injury report and the market lines.
3. The agent calls the model and gets a predicted distribution for PTS, REB and AST.
4. The agent answers in plain English: prediction, 80% range, top 3 drivers (for example, "Reaves out: +1.8 AST"), and the edge against each market strike.
5. After the game, a job logs actual stats and scores the prediction.

## Success metrics

- **Model accuracy:** MAE for PTS, REB and AST beats two baselines: season average, and the market-implied median.
- **Calibration:** actuals land inside the 80% range 80% of the time (±10 points); predicted P(≥ k) is calibrated across strikes.
- **Paper trading:** positive ROI after costs over 50+ bets, positive average CLV, max drawdown under 20% of the simulated bankroll.
- **Agent quality:** 100% of numbers in chat answers traceable to a tool call; every research note cites a source.
- **Usage:** I use it for at least 20 games this season.

## Data model (MongoDB Atlas)

| Collection | Type | Contents |
|---|---|---|
| `games` | Regular; unique `game_id` | One doc per Luka game: opponent, home/away, minutes, PTS/REB/AST, usage, rest days, back-to-back flag, embedded `teammates_out` |
| `players` | Regular | Teammates and opponents with key season stats |
| `injury_reports` | Time series | Status snapshots over time, so the model only sees what was known before tip-off |
| `market_snapshots` | Time series; meta `{venue, game_id, market_id, stat, strike}` | Kalshi and Polymarket bid, ask and volume every 15 min before tip-off |
| `news` | Atlas Vector Search (Voyage AI) | Injury reports and game recaps for retrieval |
| `features` | Regular; unique `game_id` | Exact feature values the model saw, plus an `as_of` timestamp |
| `predictions` | Regular; unique `game_id + stat + model_version` | Point prediction, range, P(≥ k) per strike, top drivers, market at prediction time, actual, error |
| `paper_trades` | Regular | `prediction_id`, strategy version, venue, ticker, side, price, stake, fees, decision timestamp, settlement, P&L |
| `strategy_configs` | Regular; versioned, never modified | Frozen bet rules |
| `agent_runs` | Regular | One doc per agent call: inputs, tool calls, output, latency, tokens, errors |
| `raw_api_cache` | Regular; TTL index | Raw API responses for retries and debugging |

### Schema design decisions (interview material)

- **Embed vs. reference:** `teammates_out` is embedded in each game because it's always read with the game. Full player profiles are referenced from `players` because they change on their own schedule.
- **When not to use time series:** `games` is small, keyed and upserted, so a regular collection with a unique index fits. Time series is for high-frequency, insert-only data (`market_snapshots`, `injury_reports`).
- **Point-in-time snapshots:** injuries, market prices and features carry timestamps, so every prediction can be replayed exactly as it was made.
- **Schema validation:** `$jsonSchema` on `predictions` and `paper_trades` so no agent can write a malformed record.
- **Absorbing source drift:** Kalshi's live and archive APIs name the same fields differently. Raw docs are cached as-is and normalized on the way in.

## MongoDB feature map

| Need | Feature |
|---|---|
| Injury and price history | Time series collections |
| Rolling 5/10-game form, opponent splits | Aggregation pipeline (`$setWindowFields`, `$group`) |
| Wake the next step when data lands | Change streams on `features` and `predictions` |
| "Find games where Reaves sat" | Atlas Vector Search with Voyage AI, plus hybrid search |
| Keep bad data out | `$jsonSchema` validation, unique indexes |
| Fast lookups | Compound indexes such as `{player_id: 1, game_date: -1}`; TTL on raw caches |
| Dashboards | Atlas Charts |
| Agent memory | `predictions`, `agent_runs`; optional LangGraph MongoDB checkpointer |
| Coding with AI | MongoDB MCP server in Claude Code |

## Model

- **Data:** `nba_api` game logs (Luka's career plus Lakers lineups).
- **Features:** projected minutes (last-10 average adjusted for back-to-backs and teammates out, never same-game minutes), generic teammates-out features, opponent defensive rating and pace, rest, home/away, rolling 5/10 form. Head-to-head history last.
- **Algorithm:** linear baseline and scikit-learn gradient boosting, one model per stat.
- **Distribution:** empirical out-of-sample residuals. P(stat ≥ k) = share of residuals > (k − 0.5 − prediction). The 80% range comes from the 10th and 90th residual percentiles.
- **Validation:** train on earlier seasons, test on later games. Never shuffle time.

## Architecture

Use LLM agents only where judgment is needed; everything else is deterministic code. Components communicate only through MongoDB collections (a blackboard design). A local `worker.py` listens to change streams and dispatches.

| Component | Type | Runs when | Reads | Writes |
|---|---|---|---|---|
| Ingestion | Python, no LLM | Nightly; every 15 min on game days | `nba_api`, injury feed, Kalshi, Polymarket | `games`, `injury_reports`, `market_snapshots` |
| Research agent | LLM | About 2 h before tip | `news`, `injury_reports` | `features` (minutes estimate, teammate status, sourced notes) |
| Prediction service | scikit-learn | New `features` doc (change stream) | `features`, `games` | `predictions` |
| Trading agent | Rules engine plus LLM rationale | New `predictions` doc (change stream) | `predictions`, `market_snapshots`, `strategy_configs` | `paper_trades` |
| Scoring job | Python, no LLM | After the game goes final | `games`, `predictions`, `paper_trades` | errors, settlements, P&L |
| Chat agent | Claude | On request | Everything, read-only | `agent_runs` |

**Chat tools:** `get_next_game`, `get_prediction`, `get_market_lines`, `search_news`, `get_track_record`, `explain_prediction`.
**Agent rules:** always show the range; never call anything a lock; cite drivers and sources; never state a number that didn't come from a tool.

## Paper trading

No real money, ever. The code never calls order endpoints.

- **Bankroll:** $1,000 simulated, flat $10 per bet. A quarter-Kelly variant runs as a separate strategy.
- **Model probability:** P(stat ≥ k) from the residual distribution.
- **Market probability:** the ask we'd actually pay (YES ask for over, NO ask for under), not the midpoint.
- **Bet rule:** edge ≥ 5 points after fees, enough volume, at most one bet per stat per game (the strike with the largest edge).
- **No lookahead:** decisions use only features and snapshots timestamped before the decision, and only markets that opened before tip-off.
- **Costs:** deduct fees and spread on every trade.
- **Frozen rules:** any rule change creates a new strategy version. Results are never mixed across versions.
- **Deterministic decisions:** plain code. The LLM only writes the rationale.
- **Metrics:** P&L, ROI, win rate, bet count, max drawdown, CLV.
- **Baselines:** always over, always under, season average against the line.
- **Backtest:** Kalshi archive, 51 Luka games (Dec 2025 to Mar 2026).

## Evaluation and observability

- Every agent call goes to `agent_runs`.
- Numbers rule: every number in a chat answer comes from a tool result. Spot-check weekly.
- Chat eval set: 20 questions with known answers, rerun after every prompt or model change.
- Research notes must cite a `news` document.
- Rolling MAE and calibration in Atlas Charts; weekly retrain; model version on every prediction.

## Stack

Python 3.11; MongoDB Atlas M0 (time series, Vector Search, change streams, Charts); Voyage AI; scikit-learn; Claude API (optionally LangGraph and the MongoDB checkpointer); Streamlit; GitHub Actions for scheduling; Claude Code with the MongoDB MCP server.

## Out of scope (v1)

Other players, real money, live in-game predictions, mobile app, multi-user auth, LLM-based ingestion.

## Open questions

- ~~Do Kalshi and Polymarket list Luka props? Historical prices?~~ Yes, see the [spike](market-api-spike.md).
- ~~Polymarket access from the US?~~ Public read endpoints work.
- Does `nba_api` work from GitHub Actions, or does ingestion have to run locally?
- Are M0 limits (storage, three search indexes) enough? The GitHub Student Pack MongoDB offer was applied for on 2026-10-28.
- How many bets before P&L means anything, and what MAE is good enough to talk about publicly?

## Risks

| Risk | Mitigation |
|---|---|
| Small sample (about 80 games a season) | Lean on calibration and CLV; report confidence intervals |
| `nba_api` is unofficial and can break | Raw cache with TTL; retries; run locally if needed |
| Lookahead bias | Timestamps everywhere; pre-tip-off filters; no same-game minutes |
| Chat agent invents numbers | Numbers-from-tools rule plus the eval set |
| Kalshi API changes | Normalize at ingestion; cache raw responses |
| Scope creep | Luka only, three stats, until v1 metrics are hit |

## Why MongoDB (talking points)

- **Messy, changing data:** injury and market formats differ by source and change (Kalshi's own live and archive APIs disagree on field names). Flexible documents absorb that without migrations.
- **One database instead of three:** time series, vector search and operational data in one place.
- **Change streams as the agent event bus.**
- **Agent memory and audit trail:** `predictions` and `agent_runs`.
- **Friction log:** notes on where Atlas onboarding slowed me down.
