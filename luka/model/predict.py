"""Live predictions: one predictions doc per stat for each upcoming Luka game, ~2 h before tip-off.

Runs inside the 15-minute market snapshot task, right after the snapshots. Each run:

1. finds games in `schedule` that tip off within PREDICT_LEAD and are missing a prediction,
2. fixes as_of = now (the decision time, before tip-off) and builds the game's features from
   games before its date, assuming whoever played the team's last game plays this one,
3. refits m2's chosen model per stat (from luka/model/cards/m2.json) on every finished game
   before this one,
4. reads the latest Kalshi ladder snapshotted at or before as_of,
5. inserts one features doc and one predictions doc per stat.

Write once (docs/AGENTS.md, risk 2): predictions are only ever inserted. The unique index on
(game_id, stat, model_version) makes a second insert fail, so a later run, or the GitHub
Actions copy of this task racing the laptop, can't replace a prediction before the game.

    python -m luka.model.predict                             # predict any game that is due
    python -m luka.model.predict --dry-run --game-id 0022600295   # print a prediction now, write nothing
"""
import argparse
import math
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pymongo.errors import BulkWriteError, DuplicateKeyError

from luka.config import PLAYER_ID, TEAM
from luka.features.pipeline import FEATURE_COLUMNS, PIPELINE_VERSION, build
from luka.model.train import MODEL_VERSION, STATS, fit_adjustment, load_card, load_dataset, p_at_least, prediction_range

PREDICT_LEAD = timedelta(hours=2)
# Don't start a prediction this close to tip-off: it must finish, with as_of before tip-off.
LAST_CALL = timedelta(minutes=5)
N_DRIVERS = 5
MIN_DRIVER_EFFECT = 0.05

STAT_LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "minutes": "Minutes"}
FEATURE_LABEL = {
    **{f"{s}_avg{n}": f"{STAT_LABEL[s]}, last {n} games" for s in STAT_LABEL for n in (5, 10)},
    **{f"{s}_season": f"{STAT_LABEL[s]}, season average" for s in STAT_LABEL},
    "rest_days": "Days of rest",
    "back_to_back": "Second night of a back-to-back",
    "home": "Home game",
    "opp_def_rtg_rel": "Opponent points allowed per 100 possessions, vs league",
    "opp_pace_rel": "Opponent pace, vs league",
    "n_rotation_out": "Rotation teammates out",
    "usage_share_out": "Shot share of teammates out",
    "top_teammate_out": "Top-usage teammate out",
}


def utcnow() -> datetime:
    """The real clock, in one place so tests can replace it."""
    return datetime.now(timezone.utc)


# ---- Which games are due --------------------------------------------------------------

def due_games(db, now: datetime) -> list[dict]:
    """Scheduled games of Luka's team tipping off in (now + LAST_CALL, now + PREDICT_LEAD]."""
    return list(db.schedule.find({
        "team": TEAM,
        "status": "scheduled",
        "tip_time_tbd": {"$ne": True},
        "tip_off": {"$gt": now + LAST_CALL, "$lte": now + PREDICT_LEAD},
    }).sort("tip_off", 1))


def missing_stats(db, game_id: str) -> list[str]:
    have = set(db.predictions.distinct("stat", {"game_id": game_id, "model_version": MODEL_VERSION}))
    return [s for s in STATS if s not in have]


# ---- Inputs ---------------------------------------------------------------------------

def input_warnings(db, game: dict, now: datetime) -> tuple[list[str], datetime | None]:
    """Is every earlier game of the team in the box scores? Returns (warnings, inputs_through).

    The daily game-log task normally loads last night's game by morning. If it hasn't run,
    the features silently describe Luka as of an older game; record that instead.
    """
    last = db.team_player_games.find_one(
        {"team": game["team"], "game_date": {"$lt": game["game_date"]}}, sort=[("game_date", -1)])
    through = last["game_date"] if last else None
    warnings = []
    missed = db.schedule.find({
        "team": game["team"], "status": {"$ne": "postponed"},
        "game_date": {"$lt": game["game_date"], **({"$gt": through} if through else {})},
        "tip_off": {"$lt": now},
    })
    for m in missed:
        warnings.append(f"no box score yet for {m['game_key']} ({m['game_id']}); inputs end at the game before it")
    return warnings, through


def live_row(db, game: dict) -> pd.Series:
    """The upcoming game's feature row, built by the same pipeline as every historical row."""
    df = build(db, PLAYER_ID, upcoming=game)
    row = df[df["game_id"] == game["game_id"]]
    if len(row) != 1:
        raise RuntimeError(f"feature pipeline returned {len(row)} rows for {game['game_id']}")
    return row.iloc[0]


def fill_missing(x: pd.Series, train: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Ridge can't take a missing value; use the training average and say so."""
    missing = [c for c in FEATURE_COLUMNS if pd.isna(x[c])]
    x = x.copy()
    for c in missing:
        x[c] = train[c].mean()
    return x, ([f"no value for {', '.join(missing)}; used the training average"] if missing else [])


def latest_ladder(db, game_key: str, stat: str, as_of: datetime) -> dict | None:
    """Strikes listed in the latest snapshot at or before as_of. Kalshi first, Polymarket if none.

    Snapshots are only taken before tip-off and as_of is before tip-off, so a strike that
    opened after tip-off can't appear here.
    """
    for venue in ("kalshi", "polymarket"):
        q = {"meta.game_key": game_key, "meta.stat": stat, "meta.venue": venue, "snapshot_at": {"$lte": as_of}}
        last = db.market_snapshots.find_one(q, {"snapshot_at": 1}, sort=[("snapshot_at", -1)])
        if last:
            rows = db.market_snapshots.find({**q, "snapshot_at": last["snapshot_at"]}, {"meta.strike": 1})
            return {"venue": venue, "snapshot_at": last["snapshot_at"], "game_key": game_key,
                    "strikes": sorted({int(r["meta"]["strike"]) for r in rows})}
    return None


# ---- The prediction -------------------------------------------------------------------

def drivers(model, x: pd.Series, train: pd.DataFrame) -> list[dict]:
    """How far each feature moves the adjustment from what an average game would get.

    effect = model(x) - model(x with that one feature set to its training average). For ridge
    this is exactly the feature's term in the linear sum; for boosting it is an approximation.
    """
    if model is None:
        return []  # the baseline makes no adjustment, so nothing drives it away from the average
    X = pd.DataFrame([x[FEATURE_COLUMNS].astype(float)])
    base = float(model.predict(X)[0])
    effects = []
    for c in FEATURE_COLUMNS:
        X2 = X.copy()
        X2[c] = train[c].mean()
        effects.append((c, base - float(model.predict(X2)[0])))
    effects.sort(key=lambda e: -abs(e[1]))
    return [{"feature": c, "label": FEATURE_LABEL.get(c, c), "value": round(float(x[c]), 3), "effect": round(e, 2)}
            for c, e in effects[:N_DRIVERS] if abs(e) >= MIN_DRIVER_EFFECT]


def strike_grid(pred: float, residuals: np.ndarray, ladder: list[int]) -> list[int]:
    """Every whole number k from 1 up to where P(stat >= k) reaches 0, plus the listed ladder.

    The whole grid, not just today's ladder, so a strike Kalshi adds later can still be scored
    and paper-traded against this prediction without recomputing it.
    """
    top = math.floor(pred + residuals.max() + 0.5) + 1
    return sorted(set(range(1, max(top, 1) + 1)) | set(ladder))


def predict_stat(stat: str, spec: dict, train: pd.DataFrame, x: pd.Series) -> dict:
    residuals = np.asarray(spec["residuals"])
    model = fit_adjustment(spec["kind"], spec["params"], train, stat)
    baseline = float(x[f"{stat}_season"])
    adj = 0.0 if model is None else float(model.predict(pd.DataFrame([x[FEATURE_COLUMNS].astype(float)]))[0])
    pred = max(baseline + adj, 0.0)
    lo, hi = prediction_range(np.array([pred]), residuals)
    return {
        "stat": stat, "baseline": round(baseline, 2), "prediction": round(pred, 2),
        "low": round(float(lo[0]), 2), "high": round(float(hi[0]), 2),
        "model": {"kind": spec["kind"], "params": spec["params"]},
        "drivers": drivers(model, x, train), "residuals": residuals,
    }


def prediction_docs(game: dict, as_of: datetime, features_id, card: dict, results: list[dict],
                    ladders: dict[str, dict | None]) -> list[dict]:
    docs = []
    for r in results:
        ladder = ladders.get(r["stat"])
        strikes = strike_grid(r["prediction"], r["residuals"], ladder["strikes"] if ladder else [])
        docs.append({
            "game_id": game["game_id"],
            "game_key": game["game_key"],
            "stat": r["stat"],
            "model_version": MODEL_VERSION,
            "features_id": features_id,
            "as_of": as_of,
            "created_at": as_of,
            "tip_off": game["tip_off"],
            "baseline": r["baseline"],
            "model": r["model"],
            "prediction": r["prediction"],
            "range": {"level": card["range_level"], "low": min(r["low"], r["prediction"]),
                      "high": max(r["high"], r["prediction"])},
            "strike_probs": [{"strike": k, "p_at_least": round(p_at_least(r["prediction"], k, r["residuals"]), 4)}
                             for k in strikes],
            "drivers": r["drivers"],
            "market": ladder,
        })
    return docs


def earliest_tip_off(db, game: dict, as_of: datetime) -> datetime:
    """The schedule's tip-off, or the venues' if they say earlier (a game moved up since the
    daily schedule refresh). Snapshots record Polymarket's start time."""
    snap = db.market_snapshots.find_one(
        {"meta.game_key": game["game_key"], "snapshot_at": {"$lte": as_of}, "tip_off": {"$ne": None}},
        {"tip_off": 1}, sort=[("snapshot_at", -1)])
    return min(game["tip_off"], snap["tip_off"]) if snap else game["tip_off"]


def predict_game(db, game: dict, stats: list[str], now: datetime, card: dict | None = None) -> tuple[dict, list[dict]]:
    """Build (features doc, predictions docs) for one game. Writes nothing."""
    card = card or load_card()
    as_of = now
    game = {**game, "tip_off": earliest_tip_off(db, game, as_of)}
    if as_of >= game["tip_off"]:
        raise ValueError(f"{game['game_id']} tipped off at {game['tip_off']}; too late to predict")

    warnings, through = input_warnings(db, game, now)
    x = live_row(db, game)
    # Train on finished games before this one. load_dataset only holds finished games, so the
    # date filter is a second guard, not the only one.
    train = load_dataset(db)
    train = train[train["game_date"] < game["game_date"]]
    x, filled = fill_missing(x, train)
    warnings += filled

    features = {
        "game_id": game["game_id"],
        "player_id": PLAYER_ID,
        "as_of": as_of,
        "computed_at": now,
        "pipeline_version": PIPELINE_VERSION,
        "availability_source": "last_box_score",
        "values": {c: round(float(x[c]), 4) for c in FEATURE_COLUMNS},
        "warnings": warnings,
    }
    if through is not None:
        features["inputs_through"] = through

    results = [predict_stat(s, card["stats"][s], train, x) for s in stats]
    for r in results:  # enough to refit the same model later
        r["model"].update(train_rows=len(train), train_through=train["game_date"].max().to_pydatetime())
    ladders = {s: latest_ladder(db, game["game_key"], s, as_of) for s in stats}
    return features, prediction_docs(game, as_of, None, card, results, ladders)


def write_once(db, features: dict, preds: list[dict]) -> int:
    """Insert the features doc and the predictions. Returns how many predictions were new.

    insert_many with ordered=True stops at the first duplicate. Predictions are inserted in
    STATS order, so if another run got there first, this one stops at the stat it already wrote.

    created_at is the moment of writing, not as_of: if the laptop slept mid-run, the gap
    shows, and a prediction that would land after tip-off is refused (here and by the validator).
    """
    written = utcnow()
    if written >= preds[0]["tip_off"]:
        raise RuntimeError(f"{preds[0]['game_id']}: run finished after tip-off; not writing")
    features["computed_at"] = written
    for p in preds:
        p["created_at"] = written
    try:
        features_id = db.features.insert_one(features).inserted_id
    except DuplicateKeyError:
        return 0  # a run with this exact as_of already wrote the features, and so the predictions
    for p in preds:
        p["features_id"] = features_id
    try:
        return len(db.predictions.insert_many(preds, ordered=True).inserted_ids)
    except BulkWriteError as e:
        if any(err["code"] != 11000 for err in e.details["writeErrors"]):
            raise
        n = e.details["nInserted"]  # the rest already existed: another run wrote them first
        if n == 0:
            db.features.delete_one({"_id": features_id})  # nothing points at it
        return n


def _summary(game: dict, features: dict, preds: list[dict]) -> str:
    lines = [f"{game['game_key']} ({game['game_id']}) tip {game['tip_off']:%Y-%m-%d %H:%MZ}  as_of {features['as_of']:%H:%MZ}"]
    for w in features["warnings"]:
        lines.append(f"  WARNING {w}")
    for p in preds:
        m = p["market"]
        ladder = f"{m['venue']} ladder {m['strikes']} @ {m['snapshot_at']:%H:%MZ}" if m else "no market snapshot"
        lines.append(f"  {p['stat']}: {p['prediction']:.1f} (80% range {p['range']['low']:.1f} to {p['range']['high']:.1f}),"
                     f" season avg {p['baseline']:.1f}, {p['model']['kind']}; {ladder}")
        for d in p["drivers"][:3]:
            lines.append(f"      {d['effect']:+.2f}  {d['label']} = {d['value']}")
    return "\n".join(lines)


def run(db, now: datetime | None = None, dry_run: bool = False, card: dict | None = None) -> bool:
    """Predict every due game that is missing a prediction. Returns False if any game failed."""
    now = now or datetime.now(timezone.utc)
    ok = True
    for game in due_games(db, now):
        stats = missing_stats(db, game["game_id"])
        if not stats:
            continue
        try:
            features, preds = predict_game(db, game, stats, now, card)
            print(_summary(game, features, preds))
            if not dry_run:
                n = write_once(db, features, preds)
                print(f"  wrote {n} prediction(s)" + ("" if n == len(preds) else "; the rest already existed"))
        except Exception as e:  # one bad game must not stop the next, or the snapshots that ran first
            print(f"  ERROR predicting {game['game_id']}: {type(e).__name__}: {e}")
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    parser.add_argument("--game-id", help="with --dry-run: predict this scheduled game now, even if not due yet")
    args = parser.parse_args()
    if args.game_id and not args.dry_run:
        parser.error("--game-id only works with --dry-run; real predictions are made ~2 h before tip-off")

    from luka.db.client import get_db

    db = get_db()
    now = datetime.now(timezone.utc)
    if args.game_id:
        game = db.schedule.find_one({"game_id": args.game_id})
        if game is None:
            sys.exit(f"{args.game_id} is not in schedule; run python -m luka.ingest.schedule")
        features, preds = predict_game(db, game, STATS, now)
        print(_summary(game, features, preds))
        sys.exit(0)
    sys.exit(0 if run(db, now, dry_run=args.dry_run) else 1)


if __name__ == "__main__":
    main()
