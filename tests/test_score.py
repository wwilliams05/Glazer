import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from luka.model.m3 import rps
from luka.model.score import paired, rps_from_strikes, run, track_record
from luka.model.train import p_at_least
from test_features import db  # noqa: F401  (fixture: empty luka_test with the schema applied)

RES = np.linspace(-8, 8, 41)


def _strikes(pred):
    top = int(pred + RES.max() + 0.5) + 1
    return [{"strike": k, "p_at_least": p_at_least(pred, k, RES)} for k in range(1, top + 1)]


@pytest.mark.parametrize("actual", [3, 25, 31, 60])
def test_rps_from_stored_strikes_matches_the_walk_forward_rps(actual):
    assert rps_from_strikes(_strikes(30.0), actual) == pytest.approx(rps(np.array([30.0]), np.array([actual]), RES))


atlas = pytest.mark.skipif(not os.environ.get("MONGODB_URI"), reason="MONGODB_URI not set")
TIP = datetime(2026, 1, 10, 3, 0, tzinfo=timezone.utc)


def _pred(game_id, stat="pts", version="m2"):
    return {"game_id": game_id, "stat": stat, "model_version": version, "as_of": TIP - timedelta(hours=2),
            "created_at": TIP - timedelta(hours=2), "tip_off": TIP, "baseline": 29.0, "prediction": 30.0,
            "range": {"level": 0.8, "low": 24.0, "high": 36.0}, "strike_probs": _strikes(30.0), "drivers": []}


def _sched(game_id, status="final", tip=TIP):
    return {"game_id": game_id, "game_key": game_id, "season": "2025-26", "season_type": "regular",
            "game_date": TIP, "tip_off": tip, "team": "AAA", "opponent": "BBB", "home": True,
            "status": status, "updated_at": TIP}


def _box(game_id, player_id, pts=10):
    return {"game_id": game_id, "player_id": player_id, "name": "P", "team": "AAA", "season": "2025-26",
            "season_type": "regular", "game_date": TIP, "minutes": 30, "pts": pts, "reb": 5, "ast": 5,
            "fga": 10, "fta": 2, "tov": 1}


@atlas
def test_scores_once_voids_dnp_and_waits_for_box_scores(db):
    db.predictions.insert_many([_pred(g) for g in ("PLAYED", "DNP", "PENDING", "LIVE", "MOVED", "LOGFAIL")]
                               + [_pred("PLAYED", version="m3")])
    db.schedule.insert_many([_sched("PLAYED"), _sched("DNP"), _sched("PENDING"), _sched("LIVE", status="live"),
                             _sched("MOVED", status="scheduled", tip=TIP + timedelta(days=20)), _sched("LOGFAIL")])
    db.games.insert_one({"game_id": "PLAYED", "player_id": 1629029, "season": "2025-26", "game_date": TIP,
                         "opponent": "BBB", "home": True, "minutes": 36, "pts": 40, "reb": 8, "ast": 9})
    db.team_player_games.insert_many([_box("DNP", 1), _box("LIVE", 1629029),
                                      _box("LOGFAIL", 1), _box("LOGFAIL", 1629029, pts=33)])
    assert run(db, now=TIP + timedelta(hours=1)) and db.predictions.count_documents({"scored_at": {"$exists": True}}) == 0
    run(db, now=TIP + timedelta(hours=10))
    played = db.predictions.find_one({"game_id": "PLAYED", "model_version": "m2"})
    assert played["actual"] == 40 and played["error"] == 10 and played["in_range"] is False
    assert played["rps"] == pytest.approx(rps_from_strikes(_strikes(30.0), 40), abs=1e-4)
    assert played["prediction"] == 30.0 and played["range"]["high"] == 36.0  # the prediction itself is untouched
    assert db.predictions.find_one({"game_id": "DNP"})["void_reason"] == "Luka didn't play"
    assert db.predictions.find_one({"game_id": "MOVED"})["void_reason"] == "game moved after the prediction"
    assert "scored_at" not in db.predictions.find_one({"game_id": "PENDING"})  # final, but no box scores yet
    assert "scored_at" not in db.predictions.find_one({"game_id": "LIVE"})  # box score present, game not final
    # Luka's own log request failed (no games row), but his team's box score has him: scored, not voided.
    assert db.predictions.find_one({"game_id": "LOGFAIL"})["actual"] == 33

    first = played["scored_at"]
    db.games.update_one({"game_id": "PLAYED"}, {"$set": {"pts": 12}})  # a stat correction later...
    run(db, now=TIP + timedelta(days=2))
    assert db.predictions.find_one({"_id": played["_id"]})["scored_at"] == first  # ...doesn't rescore

    pair = paired(db)
    assert pair["pts"]["games"] == 1 and pair["pts"]["rps_diff"] == 0  # same doc content in this fixture
    table = track_record(db)
    # m2 scored PLAYED (40 vs 30) and LOGFAIL (33 vs 30); season average was 29 for both.
    assert table["m2 pts"]["games"] == 2 and table["m2 pts"]["mae"] == 6.5 and table["m2 pts"]["small_sample"]
    assert table["m2 pts"]["baseline_mae"] == 7.5 and "m3 pts" in table
