"""Research (2026-11-25): model vs Kalshi on archived Luka props (2025-26 season, all before 2026-11-23).

For every archived Luka PTS/REB/AST strike whose market opened before tip-off, take the last
hourly candle that closed at or before tip-off. Compare Brier scores of the market's mid price
and of m2's / m3's P(stat >= k) (models trained on seasons before 2025-26). Read-only GETs.
Results are in docs/PLAN.md ("Model vs market").

    PYTHONPATH=. .venv/Scripts/python scripts/research/market_bench.py <cache folder>
"""
import json, os, sys, time
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests
from luka.db.client import get_db
from luka.ingest.markets import parse_kalshi_event
from luka.model import m3
from luka.model.train import load_dataset, load_card, fit_predict, p_at_least, TEST_FROM
K = "https://api.elections.kalshi.com/trade-api/v2"
OUT = sys.argv[1]  # a scratch folder for the cached archive and candles, e.g. a temp dir
s = requests.Session()
cache = f"{OUT}/kalshi_archive.json"
if os.path.exists(cache):
    markets = json.load(open(cache))
else:
    markets = []
    for series, stat in (("KXNBAPTS", "pts"), ("KXNBAREB", "reb"), ("KXNBAAST", "ast")):
        cursor = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **({"cursor": cursor} if cursor else {})}
            r = s.get(f"{K}/historical/markets", params=p, timeout=30).json()
            markets += [{**m, "_stat": stat} for m in r.get("markets", []) if "DONCIC" in m["ticker"]]
            cursor = r.get("cursor"); time.sleep(0.15)
            if not cursor: break
    json.dump(markets, open(cache, "w"))
print(len(markets), "archived Luka markets")
# Tip-offs for 2025-26 from the schedule endpoint (memory only).
from nba_api.stats.endpoints import scheduleleaguev2
sched = scheduleleaguev2.ScheduleLeagueV2(season="2025-26", league_id="00", timeout=30).get_normalized_dict()["SeasonGames"]
tips = {f"{g['gameDateEst'][:10]}_{g['awayTeam_teamTricode']}_{g['homeTeam_teamTricode']}":
        datetime.fromisoformat(g["gameDateTimeUTC"].replace("Z", "+00:00")) for g in sched}
db = get_db()
keys = {g["game_key"]: g["game_id"] for g in db.games.find({}, {"game_key": 1, "game_id": 1})}
CUT = datetime(2026, 11, 23, tzinfo=timezone.utc)
rows = []
cc = f"{OUT}/kalshi_candles.json"; candles = json.load(open(cc)) if os.path.exists(cc) else {}
for m in markets:
    g = parse_kalshi_event(m["event_ticker"])
    if g is None or g.key not in tips or g.key not in keys: continue
    tip = tips[g.key]; opened = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
    if opened >= tip or tip >= CUT or m.get("result") not in ("yes", "no"): continue
    if m["ticker"] not in candles:
        r = s.get(f"{K}/historical/markets/{m['ticker']}/candlesticks",
                  params={"start_ts": int(opened.timestamp()), "end_ts": int(tip.timestamp()), "period_interval": 60}, timeout=30)
        candles[m["ticker"]] = r.json().get("candlesticks", []) if r.ok else []
        time.sleep(0.12)
    cs = [c for c in candles[m["ticker"]] if c["end_period_ts"] <= tip.timestamp()]
    if not cs: continue
    c = cs[-1]
    def px(d, *names):
        for n in names:
            v = d.get(n) if isinstance(d, dict) else None
            if v not in (None, ""): return float(v)
    bid = px(c.get("yes_bid", {}), "close_dollars", "close"); ask = px(c.get("yes_ask", {}), "close_dollars", "close")
    if bid is None or ask is None: continue
    if bid > 1.5: bid, ask = bid / 100, ask / 100  # cents in archived candles
    rows.append({"game_id": keys[g.key], "stat": m["_stat"], "strike": int(float(m["floor_strike"]) + 0.5),
                 "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "y": 1.0 if m["result"] == "yes" else 0.0})
json.dump(candles, open(cc, "w"))
mk = pd.DataFrame(rows); print(len(mk), "strikes with a pre-tip quote,", mk.game_id.nunique(), "games")
# Model probabilities on these games: m2 (its card, f1) and m3 (its card, f2), trained before 2025-26.
d2 = load_dataset(db); d2 = d2[d2.game_date < CUT]; c2 = load_card()
d3 = m3.load_frame(db); c3 = m3.load_card()
preds = {}
for stat in ["pts", "reb", "ast"]:
    te2 = d2[d2.season >= TEST_FROM]
    p2 = fit_predict(c2["stats"][stat]["kind"], c2["stats"][stat]["params"], d2[d2.season < TEST_FROM], te2, stat)
    te3 = d3[d3.season >= TEST_FROM]
    p3 = m3.candidate_predict(m3.fit_candidate(tuple(c3["stats"][stat]["members"]), d3[d3.season < TEST_FROM], stat), te3, stat)
    for gid, a in zip(te2.game_id, p2): preds[("m2", gid, stat)] = a
    for gid, a in zip(te3.game_id, p3): preds[("m3", gid, stat)] = a
res = {v: {st: np.asarray(c["stats"][st]["residuals"]) for st in ["pts", "reb", "ast"]} for v, c in (("m2", c2), ("m3", c3))}
for v in ("m2", "m3"):
    mk[v] = [p_at_least(preds[(v, r.game_id, r.stat)], r.strike, res[v][r.stat]) if (v, r.game_id, r.stat) in preds else np.nan
             for r in mk.itertuples()]
mk = mk.dropna(subset=["m2", "m3"])
mk["spread"] = mk.ask - mk.bid
def brier(p, y): return np.mean((p - y) ** 2)
for name, sub in (("all", mk), ("spread <= 5c", mk[mk.spread <= 0.05])):
    print(f"\n{name}: {len(sub)} strikes, {sub.game_id.nunique()} games")
    for stat in ["pts", "reb", "ast", None]:
        q = sub if stat is None else sub[sub.stat == stat]
        if not len(q): continue
        print(f"  {stat or 'all':<4} n={len(q):>4}  Brier market {brier(q.mid, q.y):.4f}  m2 {brier(q.m2, q.y):.4f}  m3 {brier(q.m3, q.y):.4f}"
              f"   blend(market,m3) {brier((q.mid + q.m3) / 2, q.y):.4f}")

# Game-clustered bootstrap: strikes in one game share one outcome, so resample games.
rng = np.random.default_rng(0)
liq = mk[mk.spread <= 0.05].copy()
for v in ("m2", "m3"):
    liq[f"d_{v}"] = (liq.mid - liq.y) ** 2 - (liq[v] - liq.y) ** 2  # positive = model better than market
    per_game = liq.groupby("game_id")[f"d_{v}"].agg(["sum", "count"])
    games = per_game.index.to_numpy()
    boots = []
    for _ in range(4000):
        pick = per_game.loc[rng.choice(games, len(games))]
        boots.append(pick["sum"].sum() / pick["count"].sum())
    print(f"liquid strikes, market Brier minus {v} Brier: {liq[f'd_{v}'].mean():+.4f}  "
          f"(90% game-bootstrap {np.percentile(boots, 5):+.4f} to {np.percentile(boots, 95):+.4f})")
