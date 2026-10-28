# Market API spike: Kalshi and Polymarket (2026-10-28)

Question from the PRD: do Kalshi and Polymarket list Luka player props regularly, and can we get historical prices for a backtest?

**Answer: yes on both, and Kalshi is the primary benchmark.** All endpoints below are public reads with no API key, tested from a US residential connection.

## Kalshi (primary)

| | Finding |
|---|---|
| Series | `KXNBAPTS`, `KXNBAREB`, `KXNBAAST` (also `KXNBAPRA`, `KXNBAPA`, `KXNBAPR`, `KXNBARA` for combos) |
| Market shape | Threshold ladder per game: "Luka Dončić: 30+ points" (`strike_type: greater`, `floor_strike: 29.5`). Points ladder 25/30/…/55, rebounds and assists 4/6/…/14 |
| Ticker | `KXNBAPTS-26OCT27PORLAL-LALLDONCIC77-30` = series, date and teams, player, strike |
| Liquidity (Oct 27 vs POR) | PTS 25+ $144K, 30+ $110K, 40+ $139K; AST 6+ $61K; REB 4+ $21K. Real markets |
| Open time | This season about 26 hours before tip. Last season about 1 to 2 hours before tip |
| Live price history | `GET /series/{series}/markets/{ticker}/candlesticks?start_ts&end_ts&period_interval=1\|60\|1440` returns bid, ask and last per period |
| Trades | `GET /markets/trades?ticker=…` with taker side, price and size |
| Archive | `GET /historical/markets?series_ticker=…` (paginated by cursor) and `GET /historical/markets/{ticker}/candlesticks`. **51 Luka games from 2025-12-01 to 2026-03-31** |
| Settlement | `result` is yes or no. If the player is inactive, the market settles at a "fair price" rather than yes or no (a scoring edge case) |

Gotchas:
- **Field names differ between live and archived candles** (`close_dollars` vs `close`). The ingestion code has to normalize them. Good "flexible documents" material.
- **Some strikes are created mid-game** (the 14+ assists market opened at 05:13Z during the game). Only use markets and snapshots with `open_time` and snapshot time before tip-off.
- Ladders mean the model needs P(stat ≥ k) for every strike, not one line. The residual-distribution approach gives that directly.
- Base URL: `https://api.elections.kalshi.com/trade-api/v2`. **Read-only. Never call order or portfolio endpoints.**

## Polymarket (secondary)

| | Finding |
|---|---|
| Discovery | Gamma API `GET https://gamma-api.polymarket.com/events?slug=nba-por-lal-2026-10-27`. One event per game with hundreds of markets |
| Market shape | "Luka Dončić: Points O/U 32.5" plus alternate lines (9.5 … 39.5), and the same for REB and AST |
| Liquidity | Thin: about $10 to $1K per market. Treat as a comparison line, not a tradable benchmark |
| Price history | `GET https://clob.polymarket.com/prices-history?market={token_id}&startTs&endTs&fidelity=60`. **Closed markets need explicit `startTs` and `endTs`**; `interval=max` comes back empty |
| Archive | Last-season Luka props exist (for example `nba-atl-lal-2026-01-13`) and their history is retrievable |
| US access | Read endpoints worked from here |

## Decisions this drives

1. **Kalshi is the benchmark** for paper trading and "beat the market" MAE. Polymarket is logged as a second opinion.
2. **The backtest is possible** on about 51 games from last season (Kalshi) using prices from before tip-off.
3. Model output: a full predictive distribution per stat, used to price every strike on the ladder.
4. Bet rule for ladders: at most one bet per stat per game. Take the strike with the largest edge over the ask, after fees.
