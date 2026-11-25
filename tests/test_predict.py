import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from luka.features.pipeline import FEATURE_COLUMNS, PIPELINE_VERSION, build, feature_docs, write
from luka.ingest import markets
from luka.model import predict
from luka.model.predict import drivers, fill_missing, strike_grid, write_once
from luka.model import m3
from luka.model.train import MODEL_VERSION, fit_adjustment
from test_features import _load, _season, day, db  # noqa: F401  (db is a fixture; pytest puts tests/ on the path)

RESIDUALS = list(np.linspace(-8, 8, 41))  # symmetric, so the 80% range is about +-6.4


def _card(kind="ridge", params=None):
    spec = {"kind": kind, "params": params if params is not None else {"alpha": 1.0}, "residuals": RESIDUALS}
    return {"model_version": MODEL_VERSION, "pipeline_version": PIPELINE_VERSION, "range_level": 0.8,
            "stats": {s: dict(spec) for s in ("pts", "reb", "ast")}}


# ---- No database ------------------------------------------------------------------------

def test_strike_grid_covers_ladder_and_runs_to_zero():
    grid = strike_grid(30.0, np.array(RESIDUALS), ladder=[45, 60])
    assert grid[0] == 1 and 45 in grid and 60 in grid
    assert 38 in grid and 39 in grid  # 30 + max residual 8, rounded, plus one more where P = 0


def _train(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df["pts_season"] = 30.0
    df["pts"] = 30 + 2 * df["pts_avg5"] + rng.normal(scale=0.1, size=n)
    return df


def test_baseline_has_no_drivers():
    assert drivers(None, _train().iloc[0], _train()) == []


def test_ridge_drivers_are_its_linear_terms():
    train = _train()
    model = fit_adjustment("ridge", {"alpha": 0.01}, train, "pts")
    x = train.iloc[0].copy()
    x["pts_avg5"] = train["pts_avg5"].mean() + 1.5  # 1.5 above average, coefficient about 2 per standard deviation
    d = drivers(model, x, train)
    assert d[0]["feature"] == "pts_avg5" and d[0]["label"] == "Points, last 5 games"
    ridge = model[-1]
    scale = model[0].scale_[FEATURE_COLUMNS.index("pts_avg5")]
    assert d[0]["effect"] == pytest.approx(ridge.coef_[FEATURE_COLUMNS.index("pts_avg5")] * 1.5 / scale, abs=0.01)


def test_fill_missing_uses_training_average_and_says_so():
    train = _train()
    x = train.iloc[0].copy()
    x["opp_pace_rel"] = np.nan
    filled, warnings = fill_missing(x, train)
    assert filled["opp_pace_rel"] == pytest.approx(train["opp_pace_rel"].mean())
    assert "opp_pace_rel" in warnings[0]


# ---- End to end against a real cluster -----------------------------------------------------

atlas = pytest.mark.skipif(not os.environ.get("MONGODB_URI"), reason="MONGODB_URI not set")

GAME = 20  # games 0..19 are played; 20 is upcoming
TIP = day(2 * GAME) + timedelta(hours=27)  # 7pm Pacific on the game's Eastern date
KEY = "2025-12-01_BBB_AAA"  # day(40)


def _setup(db, monkeypatch):
    monkeypatch.setattr(predict, "TEAM", "AAA")
    monkeypatch.setattr(markets, "TEAM", "AAA")
    # The run "finishes" 40 s after it starts (as_of), in the fixture's 2025.
    monkeypatch.setattr(predict, "utcnow", lambda: TIP - timedelta(hours=1, minutes=50) + timedelta(seconds=40))
    _load(db, _season(GAME))
    write(db, feature_docs(build(db)))
    db.schedule.insert_one({
        "game_id": f"00225{GAME:05d}", "game_key": KEY, "season": "2025-26", "season_type": "regular",
        "game_date": day(2 * GAME), "tip_off": TIP, "team": "AAA", "opponent": "BBB", "home": True,
        "status": "scheduled", "updated_at": TIP - timedelta(days=1),
    })

    def snap(at, strike, stat="pts"):
        return {"snapshot_at": at, "meta": {"venue": "kalshi", "game_key": KEY, "market_id": f"K-{strike}",
                                            "stat": stat, "strike": strike}, "yes_ask": 0.5}
    now = TIP - timedelta(hours=1, minutes=50)
    db.market_snapshots.insert_many([
        snap(now - timedelta(minutes=30), 20),  # an older ladder
        snap(now - timedelta(minutes=5), 25), snap(now - timedelta(minutes=5), 30),  # the latest before as_of
        snap(now + timedelta(minutes=5), 99),  # taken after the decision: must not be used
    ])
    return now


@atlas
def test_prediction_is_written_once_before_tip_off(db, monkeypatch):
    now = _setup(db, monkeypatch)
    assert predict.due_games(db, TIP - timedelta(hours=3)) == []  # too early
    assert predict.run(db, now, card=_card(), versions=("m2",))

    preds = {p["stat"]: p for p in db.predictions.find({"model_version": MODEL_VERSION})}
    assert set(preds) == {"pts", "reb", "ast"}
    pts = preds["pts"]
    assert pts["as_of"] == now and pts["as_of"] < pts["created_at"] < pts["tip_off"] == TIP
    assert pts["model"]["train_rows"] > 0 and pts["model"]["train_through"] < day(2 * GAME)
    assert pts["market"]["strikes"] == [25, 30]  # latest ladder at or before as_of; not 20, not 99
    assert pts["market"]["snapshot_at"] == now - timedelta(minutes=5)
    probs = {s["strike"]: s["p_at_least"] for s in pts["strike_probs"]}
    assert {25, 30} <= set(probs) and probs[25] >= probs[30]
    assert pts["range"]["low"] <= pts["prediction"] <= pts["range"]["high"]

    feats = db.features.find_one({"_id": pts["features_id"]})
    assert feats["availability_source"] == "last_box_score" and feats["as_of"] == now
    # Training reads historical rows only, so the live row can't leak into it.
    assert db.features.count_documents({"availability_source": "box_score"}) == GAME

    # A later run, even with new data, must not touch the stored prediction.
    db.games.update_many({}, {"$inc": {"pts": 10}})
    assert predict.run(db, now + timedelta(minutes=15), card=_card(), versions=("m2",))
    again = db.predictions.find_one({"_id": pts["_id"]})
    assert again == pts
    assert db.predictions.count_documents({}) == 3 and db.features.count_documents({"availability_source": "last_box_score"}) == 1


@atlas
def test_losing_a_race_leaves_the_first_prediction(db, monkeypatch):
    now = _setup(db, monkeypatch)
    game = db.schedule.find_one()
    features, preds = predict.predict_game(db, game, ["pts", "reb", "ast"], now, _card())
    assert write_once(db, dict(features), [dict(p) for p in preds]) == 3
    # A second writer (say, the GitHub Actions copy) a few seconds later, with a different model, gets nothing in.
    later = now + timedelta(seconds=20)
    features2, preds2 = predict.predict_game(db, game, ["pts", "reb", "ast"], later, _card("baseline", {}))
    assert write_once(db, dict(features2), [dict(p) for p in preds2]) == 0
    assert {p["model"]["kind"] for p in db.predictions.find()} == {"ridge"}
    assert db.features.count_documents({"availability_source": "last_box_score"}) == 1  # loser's doc removed
    # Same as_of to the millisecond: stopped at the features doc instead.
    assert write_once(db, dict(features), [dict(p) for p in preds]) == 0


@atlas
def test_a_run_that_finishes_after_tip_off_writes_nothing(db, monkeypatch):
    now = _setup(db, monkeypatch)
    monkeypatch.setattr(predict, "utcnow", lambda: TIP + timedelta(minutes=1))  # laptop slept mid-run
    assert not predict.run(db, now, card=_card(), versions=("m2",))
    assert db.predictions.count_documents({}) == 0 and db.features.count_documents({"availability_source": "last_box_score"}) == 0


@atlas
def test_venue_tip_off_wins_when_earlier(db, monkeypatch):
    now = _setup(db, monkeypatch)
    moved = TIP - timedelta(hours=1)
    db.market_snapshots.insert_one({"snapshot_at": now - timedelta(minutes=1), "tip_off": moved,
                                    "meta": {"venue": "polymarket", "game_key": KEY, "market_id": "P", "stat": "pts", "strike": 30}})
    features, preds = predict.predict_game(db, db.schedule.find_one(), ["pts"], now, _card())
    assert preds[0]["tip_off"] == moved


@atlas
def test_markets_skip_quiet_days_only_with_a_fresh_schedule(db, monkeypatch):
    _setup(db, monkeypatch)
    fresh = TIP - timedelta(days=1)  # schedule updated_at
    assert markets.quiet_until(db, TIP - timedelta(hours=40)) == TIP  # 40 h out: skip
    assert markets.quiet_until(db, TIP - timedelta(hours=20)) is None  # within 30 h: run
    assert markets.quiet_until(db, fresh + timedelta(days=4)) is None  # stale schedule: run


def _m3_card(members=("season_gbm", "ew_ridge300")):
    return {"model_version": "m3", "pipeline_version": "f2", "range_level": 0.8,
            "stats": {s: {"members": list(members), "residuals": RESIDUALS} for s in ("pts", "reb", "ast")}}


def test_saved_m3_card_matches_the_code():
    card = m3.load_card()  # raises if the features or components changed since the card was made
    assert set(card["stats"]) == {"pts", "reb", "ast"}
    assert card["dev_cutoff"].startswith("2026-11-23")  # chosen before the first live game


@atlas
def test_m3_runs_beside_m2_without_touching_it(db, monkeypatch):
    now = _setup(db, monkeypatch)
    assert predict.run(db, now, cards={"m2": _card(), "m3": _m3_card()})
    by_version = {}
    for p in db.predictions.find():
        by_version.setdefault(p["model_version"], {})[p["stat"]] = p
    assert set(by_version) == {"m2", "m3"} and all(len(v) == 3 for v in by_version.values())
    p3 = by_version["m3"]["pts"]
    assert p3["model"]["members"] == ["season_gbm", "ew_ridge300"]
    assert p3["range"]["low"] <= p3["prediction"] <= p3["range"]["high"] and p3["as_of"] < p3["tip_off"]
    assert p3["market"]["strikes"] == [25, 30]
    f3 = db.features.find_one({"_id": p3["features_id"]})
    assert f3["pipeline_version"] == "f2" and "pts_ew10" in f3["values"]
    f2 = db.features.find_one({"_id": by_version["m2"]["pts"]["features_id"]})
    assert f2["pipeline_version"] == "f1" and "pts_ew10" not in f2["values"]  # m2's inputs unchanged


@atlas
def test_a_broken_m3_still_lets_m2_write(db, monkeypatch):
    now = _setup(db, monkeypatch)
    assert not predict.run(db, now, cards={"m2": _card(), "m3": {"stats": {}, "range_level": 0.8}})  # m3 fails
    assert {p["model_version"] for p in db.predictions.find()} == {"m2"}
    assert db.predictions.count_documents({}) == 3
