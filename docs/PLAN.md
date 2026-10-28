# Build plan

Companion to [PRD.md](PRD.md). One milestone at a time: build, test, commit.

## Week-by-week

| Week | Dates | Deliverable | Done when |
|---|---|---|---|
| 1 | Oct 28 – Nov 3 | Atlas M0 cluster; MongoDB MCP server in Claude Code; `db/schema.py` creates collections, `$jsonSchema` validators and indexes; ~~market spike~~ ([done](market-api-spike.md)); `ingest/games.py` loads Luka's career logs; **`ingest/markets.py` starts snapshotting Kalshi and Polymarket every 15 min on game days** | Season-average baseline MAE for PTS, REB and AST; snapshots accumulating |
| 2 | Nov 4 – 10 | `features/pipeline.py` (`$setWindowFields`: rolling 5/10, rest, back-to-backs, teammates out, projected minutes) writes to `features`; linear and gradient boosting models per stat; time-ordered split; residual distribution | Beats the season-average baseline on held-out games; 80% range lands within 70–90% |
| 3 | Nov 11 – 17 | Chat agent (Claude API, read-only tools) with Streamlit UI and `agent_runs` logging; 20-question eval set | Eval set passes; every number traces back to a tool call |
| 4 | Nov 18 – 24 | Scoring job, injury snapshot ingestion, Atlas Charts (MAE, calibration) | Predictions score themselves after games |
| 5 | Nov 25 – Dec 1 | `worker.py` change-stream listener; trading rules engine and versioned strategy config; `paper_trades`; CLV; baselines; **backtest on Kalshi's 51 archived games** | Simulated bets placed automatically; backtest report |
| 6 | Dec 2 – 8 | Research agent with Voyage embeddings and hybrid search over `news`; README case study; friction log | Every research note cites a source |

## Week 1 checklist

- [ ] Create an Atlas M0 cluster (free; no credits needed). Put `MONGODB_URI` in `.env`
- [ ] Connect the MongoDB MCP server to Claude Code (read-only mode first)
- [x] `requirements.txt`, `.env.example`, package skeleton
- [x] `db/schema.py`: collections, validators and indexes, safe to re-run (apply to Atlas once the cluster exists)
- [ ] `ingest/games.py`: career game logs from `nba_api`, upserted by `game_id`. Test whether it works from GitHub Actions or has to run locally
- [x] `ingest/markets.py`: Kalshi ladder and Polymarket O/U snapshots into `market_snapshots` (archive normalization comes with the week 5 backtest)
- [ ] Schedule market snapshots (GitHub Actions cron)
- [ ] Notebook or script: season-average baseline MAE

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
