"""Live predictions: one predictions doc per stat for each upcoming Luka game, ~2 h before tip-off.

Runs inside the 15-minute market snapshot task, right after the snapshots. Each run:

1. finds games in `schedule` that tip off within PREDICT_LEAD and are missing a prediction,
2. fixes as_of = now (the decision time, before tip-off) and builds the game's features from
   games before its date, assuming whoever played the team's last game plays this one,
3. refits each model version's chosen model per stat (from luka/model/cards/<version>.json)
   on every finished game before this one,
4. reads the latest Kalshi ladder snapshotted at or before as_of,
5. inserts one features doc and one predictions doc per stat, per model version.

Model versions run side by side so live games can compare them: m2 (f1 features, the week 2
model) first, then m3 (f2 features; luka/model/m3.py). Each version is tried separately, so a
failure in m3 can't cost the m2 prediction.

Write once (docs/AGENTS.md, risk 2): predictions are only ever inserted. The unique index on
(game_id, stat, model_version) makes a second insert fail, so a later run, or the GitHub
Actions copy of this task racing the laptop, can't replace a prediction before the game.

    python -m luka.model.predict                             # predict any game that is due
    python -m luka.model.predict --dry-run --game-id 0022600295   # print a prediction now, write nothing
"""
import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pymongo.errors import BulkWriteError, DuplicateKeyError

from luka.config import PLAYER_ID, TEAM
from luka.features.pipeline import EW_COLUMNS, FEATURE_COLUMNS, PIPELINE_VERSION, build
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


def missing_stats(db, game_id: str, model_version: str = MODEL_VERSION) -> list[str]:
    have = set(db.predictions.distinct("stat", {"game_id": game_id, "model_version": model_version}))
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
        warnings.append(f"no team box score yet for {m['game_key']} ({m['game_id']}); teammate inputs end before "
                        f"it, and Luka's averages may too")
    # His own game log can fail while the team's box score loads (two different requests).
    # A game he sat out has no row of his in team_player_games either, so it doesn't trigger this.
    his = db.team_player_games.find(
        {"team": game["team"], "player_id": PLAYER_ID, "game_date": {"$lt": game["game_date"]}},
        {"game_id": 1}).sort("game_date", -1).limit(5)
    for row in his:
        if db.games.find_one({"game_id": row["game_id"], "player_id": PLAYER_ID}, {"_id": 1}) is None:
            warnings.append(f"Luka's game log for {row['game_id']} isn't loaded; his averages end before it")
    return warnings, through


def live_row(db, game: dict) -> pd.Series:
    """The upcoming game's feature row, built by the same pipeline as every historical row."""
    df = build(db, PLAYER_ID, upcoming=game)
    row = df[df["game_id"] == game["game_id"]]
    if len(row) != 1:
        raise RuntimeError(f"feature pipeline returned {len(row)} rows for {game['game_id']}")
    return row.iloc[0]


def fill_missing(x: pd.Series, train: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> tuple[pd.Series, list[str]]:
    """Ridge can't take a missing value; use the training average and say so."""
    missing = [c for c in columns if pd.isna(x[c])]
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
                    ladders: dict[str, dict | None], model_version: str = MODEL_VERSION) -> list[dict]:
    docs = []
    for r in results:
        ladder = ladders.get(r["stat"])
        strikes = strike_grid(r["prediction"], r["residuals"], ladder["strikes"] if ladder else [])
        docs.append({
            "game_id": game["game_id"],
            "game_key": game["game_key"],
            "stat": r["stat"],
            "model_version": model_version,
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


def predict_game_m3(db, game: dict, stats: list[str], now: datetime, card: dict | None = None) -> tuple[dict, list[dict]]:
    """m3's (features doc, predictions docs) for one game. Writes nothing.

    Same steps and guards as predict_game; the differences are the f2 inputs (no teammate
    features, plus the recency-weighted averages) and m3's ensembles.
    """
    from luka.model import m3  # imported here: if m3 breaks, m2 (which runs first) still works

    card = card or m3.load_card()
    as_of = now
    game = {**game, "tip_off": earliest_tip_off(db, game, as_of)}
    if as_of >= game["tip_off"]:
        raise ValueError(f"{game['game_id']} tipped off at {game['tip_off']}; too late to predict")

    warnings, through = input_warnings(db, game, now)
    x = live_row(db, game)
    # load_frame(until=game date) keeps finished games before this one only.
    train = m3.load_frame(db, until=game["game_date"])
    x, filled = fill_missing(x, train, m3.X_COLUMNS)
    warnings += filled

    stored = FEATURE_COLUMNS + EW_COLUMNS  # teammate values are kept for reading, not used by m3
    features = {
        "game_id": game["game_id"],
        "player_id": PLAYER_ID,
        "as_of": as_of,
        "computed_at": now,
        "pipeline_version": m3.PIPELINE_VERSION,
        "availability_source": "last_box_score",
        "values": {c: (None if pd.isna(x[c]) else round(float(x[c]), 4)) for c in stored},
        "warnings": warnings,
    }
    if through is not None:
        features["inputs_through"] = through

    results = [m3.predict_stat(s, card["stats"][s], train, x, FEATURE_LABEL) for s in stats]
    for r in results:
        r["model"].update(train_rows=len(train), train_through=train["game_date"].max().to_pydatetime())
    ladders = {s: latest_ladder(db, game["game_key"], s, as_of) for s in stats}
    return features, prediction_docs(game, as_of, None, card, results, ladders, m3.MODEL_VERSION)


# Live model versions, in the order they run. Each: (model_version, function building its docs).
VERSIONS = {MODEL_VERSION: predict_game, "m3": predict_game_m3}


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
    version = preds[0]["model_version"] if preds else "?"
    lines = [f"{game['game_key']} ({game['game_id']}) {version} tip {game['tip_off']:%Y-%m-%d %H:%MZ}  as_of {features['as_of']:%H:%MZ}"]
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


TASK_LIMIT_S = 300  # the Windows snapshot task's limit (PT5M); also the Actions job's timeout
PREDICT_RESERVE_S = 90  # kept free for both predictions after a catch-up
CATCH_UP_MAX_S, CATCH_UP_MIN_S, CATCH_UP_MARGIN_S = 75, 20, 15
# One cutoff for everything: a laptop catch-up must *finish* before tip-off - CATCH_UP_UNTIL,
# and GitHub Actions holds back a game with missing inputs only until then. So no catch-up can
# still be writing when either copy starts predicting, and Actions keeps a 70-minute fallback
# window (about 4-5 runs) for days the laptop is off.
CATCH_UP_UNTIL = timedelta(minutes=75)
RETRY_UNTIL = CATCH_UP_UNTIL + timedelta(minutes=15)  # before this, missing inputs wait for the next run
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def in_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def catch_up_budget() -> float:
    """Seconds a catch-up may take: what is left of the task's limit after the network wait
    and the snapshots (scripts/run_logged.py sets GLAZER_TASK_STARTED), minus the reserve."""
    started = os.environ.get("GLAZER_TASK_STARTED")
    elapsed = time.time() - float(started) if started else 0.0
    return min(CATCH_UP_MAX_S, TASK_LIMIT_S - elapsed - PREDICT_RESERVE_S)


def _tail(text: str | None, n: int) -> list[str]:
    return (text or "").splitlines()[-n:]


def _load_current_season(db, timeout_s: float) -> bool:
    """Run the daily update (`luka.ingest.games --current`) in a separate process with a hard
    time limit. A separate process, not a thread: if stats.nba.com hangs, killing the process
    stops it for good, so nothing it fetched can be written after the prediction's as_of.

    On Windows the venv's python.exe is a launcher that starts the real interpreter as its own
    child, so a timeout kills the whole process tree (taskkill /T), not just the launcher.
    """
    import subprocess

    child = subprocess.Popen(
        [sys.executable, "-m", "luka.ingest.games", "--current"], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    try:
        out, err = child.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            killed = subprocess.run(["taskkill", "/T", "/F", "/PID", str(child.pid)], capture_output=True)
            if killed.returncode != 0:
                print(f"  taskkill exit {killed.returncode}; falling back to kill()")
        child.kill()  # no-op if taskkill already ended it
        try:
            out, err = child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        if child.poll() is None:
            print("  catch-up process is still running after being killed")
            return "running"
        print(f"  catch-up stopped after {timeout_s:.0f} s (stats.nba.com slow or down); last output:")
        for line in _tail(out, 4) + _tail(err, 4):
            print(f"    {line}")
        return False
    lines = (out or "").splitlines()
    for line in [l for l in lines if "ERROR" in l] + lines[-4:]:
        print(f"    {line}")
    if child.returncode != 0:
        for line in _tail(err, 6):
            print(f"    stderr: {line}")
    return child.returncode == 0


def catch_up(db, game: dict, now: datetime, load=_load_current_season) -> bool | None:
    """Load missing box scores before predicting, if an earlier game of the team has none yet.

    The daily game-log task normally loads last night's game by morning. When it doesn't (it
    was broken 2026-11-22 to 11-25), the prediction would quietly use stale inputs. This runs
    before any version fixes its as_of, so whatever it loads still predates the decision.

    Returns None if nothing was attempted, else whether the load succeeded. Not attempted in
    GitHub Actions (stats.nba.com blocks its servers) or within CATCH_UP_UNTIL of tip-off.
    """
    warnings, _ = input_warnings(db, game, now)
    if not warnings or in_github_actions():
        return None
    # Finish before the cutoff: the budget is also capped by the time left until it, less a
    # margin for the queries before the load starts and the kill afterwards.
    to_cutoff = (game["tip_off"] - CATCH_UP_UNTIL - now).total_seconds() - CATCH_UP_MARGIN_S
    if to_cutoff <= 0:
        return None
    budget = min(catch_up_budget(), to_cutoff)
    if budget < CATCH_UP_MIN_S:
        print(f"  {game['game_id']}: only {budget:.0f} s available in this run; no catch-up")
        return "no_time"
    print(f"  catching up before predicting {game['game_id']}: {len(warnings)} earlier game(s) without box scores")
    return load(db, budget)


def prepare(db, game: dict, now: datetime, missing: dict[str, list], load, clock=None) -> tuple[dict | None, bool]:
    """Before predicting a game: hold back, catch up, re-check. Returns (game or None to skip, ok).

    - GitHub Actions leaves a game with missing inputs to the laptop until CATCH_UP_UNTIL.
    - The laptop catches up only while no version has predicted yet, so m2 and m3 see the same
      inputs. After a catch-up it re-reads the game (the load refreshes the schedule, which can
      move or postpone it) and, if box scores are still missing with time to spare, waits for
      the next run rather than locking a stale prediction in.
    """
    clock = clock or utcnow
    game = {**game, "tip_off": earliest_tip_off(db, game, now)}  # a venue may know it moved earlier
    if game["tip_off"] <= now + LAST_CALL:
        print(f"  {game['game_id']}: a venue shows an earlier tip-off; too late to start a prediction")
        return None, True
    early = game["tip_off"] - now >= CATCH_UP_UNTIL
    if in_github_actions() and early and input_warnings(db, game, now)[0]:
        print(f"  {game['game_id']}: inputs incomplete; leaving it to the laptop until "
              f"{(game['tip_off'] - CATCH_UP_UNTIL):%H:%MZ}")
        return None, True
    if not all(len(v) == len(STATS) for v in missing.values()):  # some version already started
        return game, True
    try:
        result = catch_up(db, game, now, load)
    except Exception as e:
        print(f"  ERROR catching up: {type(e).__name__}: {e}")
        result = False
    if result is None:
        return game, True
    if result == "no_time":  # this run is nearly out of time: retry next run if there's room
        return (None, True) if game["tip_off"] - now >= RETRY_UNTIL else (game, True)
    if result == "running":  # couldn't stop it: its writes would land after our as_of
        print(f"  {game['game_id']}: catch-up still running; not predicting it in this run")
        return None, False
    ok = result is not False
    if not ok:
        print("  ERROR catch-up failed")
    now = clock()  # the catch-up took time; check what's left with a fresh clock
    fresh = db.schedule.find_one({"game_id": game["game_id"]})
    if fresh is not None:
        fresh = {**fresh, "tip_off": earliest_tip_off(db, fresh, now)}
    if (fresh is None or fresh["status"] != "scheduled" or fresh.get("tip_time_tbd")
            or fresh["tip_off"] <= now + LAST_CALL or fresh["tip_off"] > now + PREDICT_LEAD):
        print(f"  {game['game_id']}: schedule changed during catch-up; not predicting it in this run")
        return None, ok
    if input_warnings(db, fresh, now)[0] and fresh["tip_off"] - now >= RETRY_UNTIL:
        print(f"  {game['game_id']}: box scores still missing; trying again in the next run")
        return None, ok
    return fresh, ok


def run(db, now: datetime | None = None, dry_run: bool = False, card: dict | None = None,
        versions: tuple[str, ...] = tuple(VERSIONS), cards: dict[str, dict] | None = None,
        load=_load_current_season) -> bool:
    """Predict every due game that is missing a prediction, for each model version.
    Returns False if anything failed.

    card: m2's card (tests); cards: {model_version: card} for any version (tests);
    load: how catch_up loads data (tests).
    """
    fixed_now = now  # tests pass a time; live runs read the clock afresh for each version
    now = now or utcnow()
    cards = {**({MODEL_VERSION: card} if card else {}), **(cards or {})}
    ok = True
    for game in due_games(db, now):
        missing = {v: missing_stats(db, game["game_id"], v) for v in versions}
        if not any(missing.values()):
            continue
        if not dry_run:
            try:
                game, prep_ok = prepare(db, game, fixed_now or utcnow(), missing, load,
                                        clock=(lambda: fixed_now) if fixed_now else utcnow)
            except Exception as e:  # e.g. an Atlas blip: skip this game this run, keep the others
                print(f"  ERROR preparing {game['game_id']}: {type(e).__name__}: {e}")
                game, prep_ok = None, False
            ok &= prep_ok
            if game is None:
                continue
        for version in versions:
            stats = missing_stats(db, game["game_id"], version)
            if not stats:
                continue
            try:  # separately per version and game: one failure must not stop the others
                # as_of = when this version starts reading: m3 runs after m2 has finished, and
                # everything it reads must predate its own as_of.
                as_of = fixed_now or utcnow()
                features, preds = VERSIONS[version](db, game, stats, as_of, cards.get(version))
                print(_summary(game, features, preds))
                if not dry_run:
                    n = write_once(db, features, preds)
                    print(f"  wrote {n} prediction(s)" + ("" if n == len(preds) else "; the rest already existed"))
            except Exception as e:
                print(f"  ERROR predicting {game['game_id']} {version}: {type(e).__name__}: {e}")
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
        for build_docs in VERSIONS.values():
            features, preds = build_docs(db, game, STATS, now)
            print(_summary(game, features, preds))
        sys.exit(0)
    sys.exit(0 if run(db, dry_run=args.dry_run) else 1)  # no `now`: each version reads the clock


if __name__ == "__main__":
    main()
