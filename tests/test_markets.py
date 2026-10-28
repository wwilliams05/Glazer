from datetime import date, datetime, timezone

import pytest

from luka.ingest import markets
from luka.ingest.markets import (
    Game, at_least_strike, is_luka_pm_market, kalshi_snapshot, parse_kalshi_event, pm_snapshot, snapshot_game,
)

GAME = Game(date(2026, 10, 27), "POR", "LAL")
NOW = datetime(2026, 10, 27, 20, 0, tzinfo=timezone.utc)
TIP = datetime(2026, 10, 28, 3, 0, tzinfo=timezone.utc)

# Shapes copied from real API responses (trimmed).
KALSHI_MARKET = {
    "ticker": "KXNBAPTS-26OCT27PORLAL-LALLDONCIC77-30",
    "event_ticker": "KXNBAPTS-26OCT27PORLAL",
    "strike_type": "greater",
    "floor_strike": 29.5,
    "yes_bid_dollars": "0.6100",
    "yes_ask_dollars": "0.6300",
    "no_bid_dollars": "0.3700",
    "no_ask_dollars": "0.3900",
    "last_price_dollars": "0.6200",
    "volume_fp": "536.58",
    "open_interest_fp": "542.67",
    "open_time": "2026-10-27T04:05:55.505859Z",
}
PM_MARKET = {
    "id": "6322001",
    "question": "Luka Dončić: Points O/U 32.5",
    "sportsMarketType": "points",
    "line": 32.5,
    "outcomes": '["Yes", "No"]',
    "bestBid": 0.45,
    "bestAsk": 0.47,
    "lastTradePrice": 0.46,
    "volumeNum": 375.57,
    "liquidityNum": 1200.0,
    "closed": False,
    "createdAt": "2026-10-27T04:52:55.966297Z",
}


def test_parse_kalshi_event():
    g = parse_kalshi_event("KXNBAGAME-26OCT28LALLAC")
    assert g == Game(date(2026, 10, 28), "LAL", "LAC")
    assert g.key == "2026-10-28_LAL_LAC"
    assert g.kalshi_suffix == "26OCT28LALLAC"
    assert g.polymarket_slug == "nba-lal-lac-2026-10-28"


@pytest.mark.parametrize("ticker", ["KXNBAGAME", "KXNBAGAME-26XYZ28LALLAC", "KXNBAPTS-26OCT27PORLAL-LALLDONCIC77-30"])
def test_parse_kalshi_event_rejects_other_tickers(ticker):
    assert parse_kalshi_event(ticker) is None


@pytest.mark.parametrize("line,k", [(29.5, 30), (32.5, 33), (3.5, 4), (30.0, None)])
def test_at_least_strike(line, k):
    assert at_least_strike(line) == k


def test_kalshi_snapshot():
    doc = kalshi_snapshot(KALSHI_MARKET, "pts", GAME, NOW, TIP)
    assert doc["meta"] == {
        "venue": "kalshi", "game_key": "2026-10-27_POR_LAL",
        "market_id": "KXNBAPTS-26OCT27PORLAL-LALLDONCIC77-30", "stat": "pts", "strike": 30,
    }
    assert (doc["yes_bid"], doc["yes_ask"], doc["no_ask"]) == (0.61, 0.63, 0.39)
    assert doc["market_open_time"].tzinfo is not None
    assert doc["snapshot_at"] == NOW and doc["tip_off"] == TIP


def test_kalshi_snapshot_skips_non_threshold_markets():
    assert kalshi_snapshot({**KALSHI_MARKET, "strike_type": "between"}, "pts", GAME, NOW, TIP) is None


def test_pm_snapshot_derives_no_side_from_yes_book():
    doc = pm_snapshot(PM_MARKET, GAME, NOW, TIP)
    assert doc["meta"]["stat"] == "pts" and doc["meta"]["strike"] == 33
    assert (doc["yes_bid"], doc["yes_ask"]) == (0.45, 0.47)
    assert (doc["no_bid"], doc["no_ask"]) == (0.53, 0.55)


def test_pm_snapshot_handles_empty_book_and_closed_markets():
    doc = pm_snapshot({**PM_MARKET, "bestBid": None}, GAME, NOW, TIP)
    assert doc["no_ask"] is None
    assert pm_snapshot({**PM_MARKET, "closed": True}, GAME, NOW, TIP) is None
    assert pm_snapshot({**PM_MARKET, "sportsMarketType": "totals"}, GAME, NOW, TIP) is None


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Luka Dončić: Points O/U 32.5", True),
        ("Luka Doncic: Assists Over 8.5", True),  # last season's wording, no diacritic
        ("Brandon Miller: Points O/U 20.5", False),
        ("Lakers vs. Trail Blazers: O/U 226.5", False),
    ],
)
def test_is_luka_pm_market(question, expected):
    assert is_luka_pm_market({"question": question}) is expected


class FakeSession:
    """Routes GETs to canned payloads by URL and params."""

    def __init__(self, routes):
        self.routes = routes

    def get(self, url, params=None, timeout=None):
        key = (url, (params or {}).get("slug") or (params or {}).get("event_ticker"))
        payload = self.routes.get(key, {"markets": []})
        return type("Resp", (), {"raise_for_status": lambda self: None, "json": lambda self: payload})()


def _routes(start_time):
    pm_event = [{"startTime": start_time, "markets": [PM_MARKET, {**PM_MARKET, "question": "Deni Avdija: Points O/U 22.5"}]}]
    return {
        (f"{markets.GAMMA}/events", GAME.polymarket_slug): pm_event,
        (f"{markets.KALSHI}/markets", "KXNBAPTS-26OCT27PORLAL"): {
            "markets": [KALSHI_MARKET, {**KALSHI_MARKET, "ticker": "KXNBAPTS-26OCT27PORLAL-PORDAVDIJA8-20"}]
        },
    }


def test_snapshot_game_collects_luka_markets_from_both_venues():
    tip_off, snaps, raw, errors = snapshot_game(FakeSession(_routes("2026-10-28T03:00:00Z")), GAME, NOW)
    assert tip_off == TIP and not errors
    assert sorted(s["meta"]["venue"] for s in snaps) == ["kalshi", "polymarket"]
    assert {r["source"] for r in raw} == {"kalshi", "polymarket"}
    assert all(set(r["payload"][0]) <= set(markets.KALSHI_RAW_FIELDS + markets.PM_RAW_FIELDS) for r in raw)


def test_snapshot_game_skips_after_tip_off():
    _, snaps, raw, _ = snapshot_game(FakeSession(_routes("2026-10-27T19:00:00Z")), GAME, NOW)
    assert snaps == [] and raw == []
