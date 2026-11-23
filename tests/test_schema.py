import copy
import os
from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError, WriteError

from luka.db.schema import COLLECTIONS, ensure_schema

load_dotenv()

PRD_COLLECTIONS = {
    "games", "players", "team_games", "team_player_games", "injury_reports", "market_snapshots", "news", "features",
    "predictions", "paper_trades", "strategy_configs", "agent_runs", "raw_api_cache",
}


def test_all_prd_collections_defined():
    assert set(COLLECTIONS) == PRD_COLLECTIONS


def test_time_series_collections_have_no_unique_indexes():
    for name, spec in COLLECTIONS.items():
        if "timeseries" in spec:
            assert not any(opts.get("unique") for _, opts in spec["indexes"]), name


def test_games_is_regular_with_unique_game_id():
    games = COLLECTIONS["games"]
    assert "timeseries" not in games
    assert ([("game_id", 1)], {"unique": True}) in games["indexes"]


def test_dry_run_plans_without_a_database():
    actions = ensure_schema(None, dry_run=True)
    assert "create market_snapshots (time series)" in actions
    assert "create games" in actions


# ---- Against a real cluster: validators must actually reject bad writes ----

atlas = pytest.mark.skipif(not os.environ.get("MONGODB_URI"), reason="MONGODB_URI not set")

T0 = datetime(2026, 10, 27, 23, 0, tzinfo=timezone.utc)

PREDICTION = {
    "game_id": "TEST001",
    "stat": "pts",
    "model_version": "test",
    "as_of": T0,
    "created_at": T0,
    "prediction": 31.5,
    "range": {"level": 0.8, "low": 24.0, "high": 39.0},
    "strike_probs": [{"strike": 30, "p_at_least": 0.58}],
    "drivers": [{"feature": "usage_share_out", "effect": 1.2}],
    "market": {"venue": "kalshi", "snapshot_at": T0 - timedelta(minutes=15)},
}

TRADE = {
    "prediction_id": ObjectId(),
    "game_id": "TEST001",
    "stat": "pts",
    "strategy_version": "test",
    "venue": "kalshi",
    "market_id": "KXNBAPTS-TEST-30",
    "strike": 30,
    "side": "yes",
    "price": 0.52,
    "stake": 10,
    "fees": 0.17,
    "snapshot_at": T0 - timedelta(minutes=15),
    "decided_at": T0,
    "tip_off": T0 + timedelta(hours=3),
    "status": "open",
}


def _with(doc, **changes):
    d = copy.deepcopy(doc)
    d.update(changes)
    return d


@pytest.fixture(scope="module")
def db():
    from luka.db.client import get_client

    name = "luka_test"
    get_client().drop_database(name)
    test_db = get_client()[name]
    ensure_schema(test_db)
    ensure_schema(test_db)  # re-running must be safe
    yield test_db
    get_client().drop_database(name)


@atlas
def test_valid_prediction_and_trade_insert(db):
    db.predictions.insert_one(copy.deepcopy(PREDICTION))
    db.paper_trades.insert_one(copy.deepcopy(TRADE))


@atlas
@pytest.mark.parametrize(
    "bad",
    [
        _with(PREDICTION, stat="stl"),
        _with(PREDICTION, prediction=45.0),  # outside its own range
        _with(PREDICTION, market={"venue": "kalshi", "snapshot_at": T0 + timedelta(minutes=1)}),  # lookahead
        {k: v for k, v in PREDICTION.items() if k != "drivers"},
    ],
    ids=["bad-stat", "outside-range", "future-market-price", "missing-drivers"],
)
def test_bad_prediction_rejected(db, bad):
    with pytest.raises(WriteError):
        db.predictions.insert_one(_with(bad, model_version="bad"))


@atlas
@pytest.mark.parametrize(
    "bad",
    [
        _with(TRADE, decided_at=T0 + timedelta(hours=4)),  # after tip-off
        _with(TRADE, snapshot_at=T0 + timedelta(minutes=5)),  # price from the future
        _with(TRADE, price=1.0),
        _with(TRADE, side="over"),
    ],
    ids=["after-tip-off", "future-snapshot", "price-1", "bad-side"],
)
def test_bad_trade_rejected(db, bad):
    with pytest.raises(WriteError):
        db.paper_trades.insert_one(_with(bad, strategy_version="bad"))


@atlas
def test_one_bet_per_stat_per_game_per_strategy(db):
    db.paper_trades.insert_one(_with(TRADE, strategy_version="dup"))
    with pytest.raises(DuplicateKeyError):
        db.paper_trades.insert_one(_with(TRADE, strategy_version="dup", strike=35))
