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
- [x] 2026-11-23 (UTA): first live `m2` prediction written at 01:09 UTC, 1 h 50 min before tip, against a Kalshi snapshot 6 s older
- [ ] The "Glazer game logs" task still points at the deleted `scripts/update_games.cmd`, so it has failed daily since 2026-11-22 (caught up by hand on 2026-11-25). Re-register it with `run_logged.py` (command in the 2026-11-25 session notes)

## m3 (added 2026-11-25): runs beside m2 on live games

Built from games before 2026-11-23 only (`DEV_CUTOFF` in `model/m3.py`; card frozen 2026-11-25). m3's clean holdout is the games it predicts live, from its first live run on (2026-11-27 SAS); m2's starts at 2026-11-23. Never backfill a prediction for a game already played. The candidate list was shaped while the 2025-26 test numbers were visible, so m3's test columns below are optimistic. Exploration scripts were throwaway; what they showed:

- **Baseline:** a recency-weighted average over Luka's earlier games (10-game half-life, across seasons; f2's `*_ew10`) beat the season-to-date average in the walk-forward for all three stats (PTS 6.62 → 6.44, REB 2.435 → 2.408, AST 2.661 → 2.602).
- **Teammates-out features added nothing,** even with perfect (box-score) knowledge of who sat, in ridge or boosting. m3 drops them, which also removes m2's calibration leak (m2's residuals were measured with same-game availability that live rows don't have).
- **More features hurt boosting** (about 400 training games); ridge was neutral. Each model family keeps a small input set.
- **Distribution shape barely matters:** empirical residuals, scaled residuals, normal and negative binomial were within 1% on RPS; Poisson was too narrow for PTS (61% coverage). m3 keeps m2's empirical residuals.
- **Single models disagreed** between walk-forward and test; an equal-weight ensemble of one model from each family was best or near best on both.

Chosen by walk-forward MAE (2021-22 to 2024-25):

| Stat | m3 | Walk-forward MAE m2 → m3 | Test MAE m2 → m3 | Test RPS m2 → m3 | Test 80% coverage m2 → m3 |
|---|---|---|---|---|---|
| PTS | season + boosting, averaged with EW10 + ridge | 6.48 → 6.34 | 7.26 → 7.31 | 5.23 → 5.23 | 77% → 77% |
| REB | average of season average and EW10 | 2.435 → 2.406 | 2.57 → 2.53 | 1.79 → 1.73 | 76% → 84% |
| AST | EW10 | 2.658 → 2.602 | 2.43 → 2.41 | 1.67 → 1.71 | 84% → 81% |

RPS = ranked probability score, the squared error of P(stat ≥ k) summed over every k; lower is better, and it rewards probabilities that are both calibrated and sharp. Test = 2025-26 to 2026-11-22 (79 games, not clean). **m3 is not proven better.** It wins the larger walk-forward by 1–2% and is mixed on the test games. Live games decide.

**Decision rule, written 2026-11-25 before any m3 live result:** after 30 live games where both versions predicted before tip-off, compare per stat on mean RPS (primary) and MAE (secondary), using the paired difference with a 90% bootstrap interval over games. A version replaces the other for paper trading on a stat only if its RPS interval excludes 0 in its favour; otherwise keep m2 as the trading model and keep both running. The winner's live numbers from those 30 games are then biased upward; report its later games separately.

- [x] `features/pipeline.py`: f2 = f1 + `*_ew10` (pandas, on the same frame as the live placeholder)
- [x] `model/m3.py`, `model/cards/m3.json` (frozen); `python -m luka.model.m3` prints the candidate table
- [x] `model/predict.py` writes m2 then m3 for each game, each in its own try
- [x] Tried and dropped: **pooling other stars' games.** A model of "stat − own EW10" trained on 17,600 star player-games (EW10 ≥ 30 min and ≥ 20 pts, 2018-19 to 2024-25) learned sensible but tiny effects (home +0.27 pts, back-to-back −0.39, +0.24 pts per possession of opponent pace). Applied to Luka's walk-forward it didn't help (PTS 6.71 → 6.78, REB 2.427 → 2.422, AST 2.645 → 2.653). Pre-game context barely moves his numbers; the game-to-game noise dominates
- [ ] Not tried yet: opponent rebounds and assists allowed (needs those columns in `team_games`), injury reports (week 4), market prices as an input
- [x] `model/score.py` (pulled forward from week 4): after each game, sets `actual`, `error`, `in_range`, `rps`, or `void` if Luka sat. Runs inside `games --current`; `python -m luka.model.score --report` prints the live track record per model version and the paired m2-vs-m3 comparison. Waits for `schedule` to say final; voids postponed or moved games. Unchecked: whether stats.nba.com files a Cup final (game id 006…) under a season type `games.py` requests; if not, such a prediction waits forever

## Model vs market (research, 2026-11-25; early input for the week 5 backtest)

`scripts/research/market_bench.py`: every archived Kalshi Luka strike (51 games, 2025-12-01 to 2026-03-31), last hourly quote at or before tip-off, settled yes or no. Brier score = mean (probability − outcome)², lower is better. Models trained on seasons before 2025-26.

| Strikes | n | Kalshi mid | m2 | m3 |
|---|---|---|---|---|
| All with a pre-tip quote | 651 (51 games) | 0.204 | 0.187 | 0.188 |
| Spread ≤ 5¢ (liquid) | 297 (41 games) | 0.186 | 0.183 | 0.183 |

- The "all strikes" gap is mostly wide-spread quotes whose mid isn't a real opinion (bid 0, ask 0.99 → 0.50).
- **On liquid strikes the model is roughly even with the market:** market minus model Brier +0.003 (90% interval over games −0.005 to +0.012 for m3; similar for m2). Not a demonstrated edge, and optimistic because m2 and m3 were designed with these games' aggregate numbers visible.
- **Paper trades pay the ask plus fees,** so parity with the mid loses money on average. Any edge has to come from strikes where model and price disagree by more than the spread plus fees; the week 5 backtest should measure exactly that, with the strategy rule fixed before it is run.

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
