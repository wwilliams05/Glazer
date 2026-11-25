"""Score stored predictions once their game is final: actual, error, in-range and RPS.

Runs daily right after the game logs load (inside `luka.ingest.games --current`). For each
prediction not yet scored whose game tipped off more than SETTLE_AFTER ago and which the
schedule marks final:

- Luka's box score is in `games` (or, if that request failed, his row in his team's box
  score, `team_player_games`) -> set actual, error (actual - prediction), in_range (inside the
  stored 80% range) and rps (below), plus scored_at.
- his team's box score is in `team_player_games` without him -> he didn't play: void (Kalshi
  settles such props at a fair price, so there is nothing to score). Decided from the team's
  own player list, so a failed request for Luka's log alone can't void a game he played.
- the game was postponed or moved by more than MOVED_BY after the prediction -> void: the
  prediction was made for a different date and inputs.
- otherwise -> leave it for the next run.

Each prediction is scored once: the update matches only docs without scored_at, and touches
only the scoring fields. The prediction, range and probabilities are never changed.

RPS (ranked probability score) = sum over k >= 1 of (P(stat >= k) - [actual >= k])^2, from the
stored strike_probs. They cover every whole k up to where P reaches 0, so the sum is the
same one luka.model.m3.rps computes in the walk-forward.

    python -m luka.model.score              # score what's ready (prints counts only)
    python -m luka.model.score --report     # track record per model version
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from luka.config import PLAYER_ID

SETTLE_AFTER = timedelta(hours=4)  # a game plus overtime is over by then
MOVED_BY = timedelta(hours=12)
SMALL_SAMPLE = 30


def rps_from_strikes(strike_probs: list[dict], actual: float) -> float:
    """Ranked probability score of one prediction. P is 0 above the stored grid."""
    probs = {int(s["strike"]): s["p_at_least"] for s in strike_probs}
    top = max(max(probs, default=0), int(actual))
    return float(sum((probs.get(k, 0.0) - (actual >= k)) ** 2 for k in range(1, top + 1)))


def score_fields(pred: dict, actual: float, now: datetime) -> dict:
    return {
        "actual": actual,
        "error": round(actual - pred["prediction"], 2),
        "in_range": bool(pred["range"]["low"] <= actual <= pred["range"]["high"]),
        "rps": round(rps_from_strikes(pred["strike_probs"], actual), 4),
        "scored_at": now,
    }


def run(db, now: datetime | None = None) -> bool:
    """Score every ready prediction. Prints counts only, never results."""
    now = now or datetime.now(timezone.utc)
    pending = list(db.predictions.find({"scored_at": {"$exists": False}, "tip_off": {"$lt": now - SETTLE_AFTER}}))
    scored = void = waiting = 0
    for p in pending:
        sched = db.schedule.find_one({"game_id": p["game_id"]}, {"status": 1, "tip_off": 1})
        if sched is not None and (sched["status"] == "postponed" or abs(sched["tip_off"] - p["tip_off"]) > MOVED_BY):
            update = {"void": True, "void_reason": "game moved after the prediction", "scored_at": now}
            void += 1
        elif sched is None or sched["status"] != "final":
            waiting += 1
            continue
        elif (row := (db.games.find_one({"game_id": p["game_id"], "player_id": PLAYER_ID}, {p["stat"]: 1})
                      or db.team_player_games.find_one({"game_id": p["game_id"], "player_id": PLAYER_ID},
                                                       {p["stat"]: 1}))) is not None:
            update = score_fields(p, float(row[p["stat"]]), now)
            scored += 1
        elif db.team_player_games.find_one({"game_id": p["game_id"]}, {"_id": 1}):
            update = {"void": True, "void_reason": "Luka didn't play", "scored_at": now}
            void += 1
        else:
            waiting += 1
            continue
        db.predictions.update_one({"_id": p["_id"], "scored_at": {"$exists": False}}, {"$set": update})
    print(f"scoring: {scored} scored, {void} void, {waiting} waiting (game not final or box scores not loaded)")
    return True


def track_record(db, model_version: str | None = None) -> dict[str, dict]:
    """Per model version and stat: games, MAE, baseline MAE, 80% coverage, mean RPS."""
    q = {"scored_at": {"$exists": True}, "void": {"$ne": True}}
    if model_version:
        q["model_version"] = model_version
    out: dict[str, dict] = {}
    for p in db.predictions.find(q):
        out.setdefault(p["model_version"], {}).setdefault(p["stat"], []).append(p)
    table = {}
    for version, by_stat in out.items():
        for stat, ps in by_stat.items():
            err = np.array([p["error"] for p in ps])
            base = np.array([p["actual"] - p["baseline"] for p in ps if p.get("baseline") is not None])
            table[f"{version} {stat}"] = {
                "games": len(ps), "mae": float(np.mean(np.abs(err))),
                "baseline_mae": float(np.mean(np.abs(base))) if len(base) else None,
                "coverage": float(np.mean([p["in_range"] for p in ps])),
                "rps": float(np.mean([p["rps"] for p in ps])),
                "small_sample": len(ps) < SMALL_SAMPLE,
            }
    return table


def paired(db, a: str = "m2", b: str = "m3") -> dict[str, dict]:
    """Version a vs b on the (game, stat) pairs both scored: the comparison PLAN.md's decision
    rule uses. Differences are a - b, so positive means b did better.
    """
    q = {"scored_at": {"$exists": True}, "void": {"$ne": True}, "model_version": {"$in": [a, b]}}
    docs: dict[tuple, dict] = {}
    for p in db.predictions.find(q, {"game_id": 1, "stat": 1, "model_version": 1, "error": 1, "rps": 1}):
        docs.setdefault((p["game_id"], p["stat"]), {})[p["model_version"]] = p
    out = {}
    rng = np.random.default_rng(0)
    for stat in sorted({k[1] for k in docs}):
        both = [v for (g, s), v in docs.items() if s == stat and a in v and b in v]
        if not both:
            continue
        d_rps = np.array([v[a]["rps"] - v[b]["rps"] for v in both])
        d_mae = np.array([abs(v[a]["error"]) - abs(v[b]["error"]) for v in both])
        boots = [d_rps[rng.integers(0, len(d_rps), len(d_rps))].mean() for _ in range(2000)]
        out[stat] = {"games": len(both), "rps_diff": float(d_rps.mean()), "mae_diff": float(d_mae.mean()),
                     "rps_diff_90": (float(np.percentile(boots, 5)), float(np.percentile(boots, 95))),
                     "small_sample": len(both) < SMALL_SAMPLE}
    return out


CALIBRATION_BINS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0001]
# Walk-forward reliability of m3 on ladder strikes (2026-11-25, PLAN.md): within 1-3 points per bin.


def calibration(db, model_version: str) -> list[dict]:
    """Reliability on the strikes the market listed: per probability bin, how often it happened.

    Uses market.strikes (the ladder at prediction time), not the whole stored grid, whose far
    strikes are near-certain and would make any model look calibrated. Well calibrated means
    `happened` is close to `predicted` in every bin, given enough strikes.
    """
    pairs = []
    q = {"model_version": model_version, "scored_at": {"$exists": True}, "void": {"$ne": True}}
    for p in db.predictions.find(q, {"strike_probs": 1, "market": 1, "actual": 1}):
        ladder = set((p.get("market") or {}).get("strikes", []))
        pairs += [(s["p_at_least"], float(p["actual"] >= s["strike"])) for s in p["strike_probs"] if s["strike"] in ladder]
    rows = []
    for lo, hi in zip(CALIBRATION_BINS, CALIBRATION_BINS[1:]):
        inside = [(pr, y) for pr, y in pairs if lo <= pr < hi]
        if inside:
            rows.append({"bin": f"{lo:.0%}-{min(hi, 1):.0%}", "strikes": len(inside),
                         "predicted": float(np.mean([pr for pr, _ in inside])),
                         "happened": float(np.mean([y for _, y in inside]))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="print the track record per model version")
    args = parser.parse_args()

    from luka.db.client import get_db

    db = get_db()
    if not args.report:
        sys.exit(0 if run(db) else 1)
    print("Each version on all of its own scored games (unpaired; different game sets):")
    for key, r in sorted(track_record(db).items()):
        base = f"{r['baseline_mae']:.2f}" if r["baseline_mae"] is not None else "-"
        flag = "  (small sample)" if r["small_sample"] else ""
        print(f"  {key:<8} {r['games']:>3} games  MAE {r['mae']:.2f} (season avg {base})  "
              f"80% coverage {r['coverage']:.0%}  RPS {r['rps']:.3f}{flag}")
    print("\nm2 vs m3 on games both predicted (m2 minus m3; positive = m3 better):")
    for stat, r in paired(db).items():
        lo, hi = r["rps_diff_90"]
        flag = "  (small sample)" if r["small_sample"] else ""
        print(f"  {stat}: {r['games']} games  RPS {r['rps_diff']:+.3f} (90% {lo:+.3f} to {hi:+.3f})  "
              f"MAE {r['mae_diff']:+.2f}{flag}")
    for version in ("m2", "m3"):
        rows = calibration(db, version)
        if rows:
            print(f"\n{version} calibration on listed strikes (all stats pooled; needs ~100+ strikes per bin to mean much):")
            for r in rows:
                print(f"  P {r['bin']:<8} {r['strikes']:>4} strikes  predicted {r['predicted']:.0%}  happened {r['happened']:.0%}")


if __name__ == "__main__":
    main()
