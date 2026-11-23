# Build plan

Companion to [PRD.md](PRD.md). One milestone at a time: build, test, commit. Weeks 2–6 were moved back two weeks on 2026-11-17 after week 1 ran long.

## Week-by-week

| Week | Dates | Deliverable | Done when |
|---|---|---|---|
| 1 | Oct 28 – Nov 17 (done) | Atlas M0 cluster; MongoDB MCP server in Claude Code; `db/schema.py` creates collections, `$jsonSchema` validators and indexes; ~~market spike~~ ([done](market-api-spike.md)); `ingest/games.py` loads Luka's career logs; **`ingest/markets.py` starts snapshotting Kalshi and Polymarket every 15 min on game days** | Season-average baseline MAE for PTS, REB and AST; snapshots accumulating |
| 2 | Nov 18 – 24 | `features/pipeline.py` (`$setWindowFields`: rolling 5/10, rest, back-to-backs, teammates out, projected minutes) writes to `features`; linear and gradient boosting models per stat; time-ordered split; residual distribution | Beats the season-average baseline on held-out games; 80% range lands within 70–90% |
| 3 | Nov 25 – Dec 1 | Chat agent (Claude API, read-only tools) with Streamlit UI and `agent_runs` logging; 20-question eval set | Eval set passes; every number traces back to a tool call |
| 4 | Dec 2 – 8 | Scoring job, injury snapshot ingestion, Atlas Charts (MAE, calibration) | Predictions score themselves after games |
| 5 | Dec 9 – 15 | `worker.py` change-stream listener; trading rules engine and versioned strategy config; `paper_trades`; CLV; baselines; **backtest on Kalshi's 51 archived games** | Simulated bets placed automatically; backtest report |
| 6 | Dec 16 – 22 | Research agent with Voyage embeddings and hybrid search over `news`; README case study; friction log | Every research note cites a source |

## Week 1 checklist

- [x] Create an Atlas M0 cluster (free; no credits needed). Put `MONGODB_URI` in `.env` (cluster `Glazer`, AWS us-east-1, schema applied 2026-11-16)
- [ ] Connect the MongoDB MCP server to Claude Code (read-only mode first)
- [x] `requirements.txt`, `.env.example`, package skeleton
- [x] `db/schema.py`: collections, validators and indexes, safe to re-run (apply to Atlas once the cluster exists)
- [x] `ingest/games.py`: career game logs from `nba_api`, upserted by `game_id` (581 games loaded 2026-11-16). GitHub Actions run failed on 2026-11-17 after ~5 min of timeouts (stats.nba.com appears to block GitHub's IPs), so the daily `--current` update runs locally: a Windows scheduled task runs `pythonw scripts/run_logged.py games_update.log luka.ingest.games --current`
- [x] `ingest/markets.py`: Kalshi ladder and Polymarket O/U snapshots into `market_snapshots` (archive normalization comes with the week 5 backtest)
- [x] Schedule market snapshots: `.github/workflows/market-snapshots.yml`, every 15 min, needs repo secret `MONGODB_URI` and Atlas Network Access `0.0.0.0/0`. Runs only from `main`
- [x] Script: season-average baseline MAE (`python -m luka.model.baseline`). As of 2026-11-16: PTS 6.63, REB 2.52, AST 2.65 over 580 games; 2025-26 on: PTS 7.37, REB 2.60, AST 2.53 (76 games)

## Week 2 checklist

- [x] `ingest/teams.py`: league-wide `team_games` (opponent defense, pace) and `team_player_games` for Luka's teams (teammates out). Refreshed daily by `games --current`
- [x] `features/pipeline.py` (version `f1`): rolling 5/10 and season-to-date form, rest, back-to-backs, home, opponent defensive rating and pace relative to the league, rotation teammates out and their share of shot volume. Windows end before the game; a test proves later games never change earlier features. Historical rows use who actually played (`availability_source: box_score`) as a stand-in for the injury report
- [x] `model/train.py` (version `m2`): predicts an adjustment to the season-average baseline; ridge, gradient boosting and "no adjustment" compete on a walk-forward over 2021-22 to 2024-25; residuals from the same walk-forward give the 80% range and P(stat ≥ k)
- [ ] Projected minutes as its own model input (deferred: minutes averages are features, but no separate projection yet)

**Results (2026-11-22, test = 2025-26 onward, 79 games):**

| Stat | Chosen | Test MAE | Baseline | Gain (90% CI) | 80% range coverage |
|---|---|---|---|---|---|
| PTS | gradient boosting | 7.26 | 7.42 | +0.16 (−0.27 to +0.60) | 77% |
| REB | baseline | 2.57 | 2.57 | 0 | 76% |
| AST | ridge (≈ baseline) | 2.43 | 2.46 | +0.03 (−0.01 to +0.07) | 84% |

Range coverage meets the 70–90% target for all three. Beating the baseline is **not proven**: the PTS gain has the right sign in both the walk-forward and the test, but its interval includes zero.

**Holdout caveat:** the first version (`m1`: raw-stat target, models chosen on 2024-25 alone) lost to the baseline on the test games (PTS 7.95, REB 2.98, AST 2.41). `m2` was designed after seeing that, so the 2025-26+ games are no longer a clean holdout. The clean check is live games from 2026-11-23 on, scored by the week 4 scoring job.

## Live predictions (added 2026-11-23, so the clean holdout gets scored)

- [x] `ingest/schedule.py`: Lakers schedule from stats.nba.com `scheduleleaguev2` into `schedule` (game_id, tip-off, status), refreshed daily by `games --current` (AGENTS.md risk 3)
- [x] `model/predict.py`: inside the 15-minute snapshot task, ~2 h before each tip-off: one features doc (`availability_source: last_box_score`, i.e. whoever played the last game) and one `m2` prediction per stat, insert-only (risk 2). P(stat ≥ k) for every whole k, not just the listed ladder; `market` records the ladder and snapshot used
- [x] `model/cards/m2.json`: the walk-forward's model choice and residuals, frozen. Live `m2` refits that choice on every finished game before the one it predicts (the test-table numbers above were trained on seasons before 2025-26 only)
- [x] `ingest/markets.py` exits early when the schedule shows no game within 30 h; falls back to the schedule's tip-off when Polymarket has none
- [ ] Laptop asleep at prediction time = no prediction (risk 6). GitHub Actions runs the same step once this is on `main`; the unique index stops the two from writing twice

## Repo layout

```
luka/
  db/          schema.py, client.py
  ingest/      games.py, markets.py, injuries.py, news.py
  features/    pipeline.py
  model/       train.py, predict.py, residuals.py
  trading/     rules.py, strategy_configs/, settle.py
  agents/      chat.py, research.py, tools.py
  worker.py
app.py
evals/
docs/
.github/workflows/
```
