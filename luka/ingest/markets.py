"""Snapshot Kalshi and Polymarket prices for Luka's PTS/REB/AST props into market_snapshots.

Meant to run every 15 minutes. Each run:
1. finds upcoming Lakers games from Kalshi's game-winner series (KXNBAGAME),
2. skips games that have tipped off (tip-off time comes from the Polymarket event),
3. writes one snapshot per open Luka prop market on each venue, plus a trimmed raw copy
   to raw_api_cache.

Strikes are normalized to "at least k" on both venues: Kalshi "30+" (floor_strike 29.5)
and Polymarket "O/U 29.5" both become k=30, so the model only has to price P(stat >= k).

    python -m luka.ingest.markets [--dry-run]
"""
import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from luka.config import KALSHI_PLAYER_TAG, PLAYER_NAME, TEAM

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"

KALSHI_SERIES = {"pts": "KXNBAPTS", "reb": "KXNBAREB", "ast": "KXNBAAST"}
KALSHI_GAME_SERIES = "KXNBAGAME"
PM_STATS = {"points": "pts", "rebounds": "reb", "assists": "ast"}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Raw cache keeps only these fields; full payloads would fill the M0 tier's 512 MB quickly.
KALSHI_RAW_FIELDS = [
    "ticker", "event_ticker", "title", "status", "strike_type", "floor_strike", "yes_bid_dollars",
    "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars", "last_price_dollars", "volume_fp",
    "open_interest_fp", "open_time", "close_time",
]
PM_RAW_FIELDS = [
    "id", "conditionId", "question", "sportsMarketType", "line", "outcomes", "bestBid", "bestAsk",
    "lastTradePrice", "volumeNum", "liquidityNum", "closed", "acceptingOrders", "createdAt",
]


@dataclass(frozen=True)
class Game:
    game_date: date  # US Eastern date, as both venues label games
    away: str
    home: str

    @property
    def key(self) -> str:
        """Our venue-neutral game id, e.g. 2026-10-28_LAL_LAC (away, then home)."""
        return f"{self.game_date.isoformat()}_{self.away}_{self.home}"

    @property
    def kalshi_suffix(self) -> str:
        d = self.game_date
        return f"{d:%y}{MONTHS[d.month - 1]}{d:%d}{self.away}{self.home}"

    @property
    def polymarket_slug(self) -> str:
        return f"nba-{self.away.lower()}-{self.home.lower()}-{self.game_date.isoformat()}"


def parse_kalshi_event(event_ticker: str) -> Game | None:
    """KXNBAGAME-26OCT28LALLAC -> Game(2026-10-28, away=LAL, home=LAC)."""
    m = re.fullmatch(r"[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})", event_ticker)
    if not m or m.group(2) not in MONTHS:
        return None
    yy, mon, dd, away, home = m.groups()
    return Game(date(2000 + int(yy), MONTHS.index(mon) + 1, int(dd)), away, home)


def at_least_strike(line: float) -> int | None:
    """Over 29.5 means at least 30. Whole-number lines can push, so we don't map them."""
    if line % 1 != 0.5:
        return None
    return math.floor(line) + 1


def _num(x) -> float | None:
    return None if x is None or x == "" else round(float(x), 4)


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _trim(market: dict, fields: list[str]) -> dict:
    return {k: market.get(k) for k in fields}


def kalshi_snapshot(market: dict, stat: str, game: Game, now: datetime, tip_off: datetime | None) -> dict | None:
    if market.get("strike_type") != "greater":
        return None
    line = float(market["floor_strike"])
    strike = at_least_strike(line)
    if strike is None:
        return None
    return {
        "snapshot_at": now,
        "meta": {"venue": "kalshi", "game_key": game.key, "market_id": market["ticker"], "stat": stat, "strike": strike},
        "line": line,
        "yes_bid": _num(market.get("yes_bid_dollars")),
        "yes_ask": _num(market.get("yes_ask_dollars")),
        "no_bid": _num(market.get("no_bid_dollars")),
        "no_ask": _num(market.get("no_ask_dollars")),
        "last": _num(market.get("last_price_dollars")),
        "volume": _num(market.get("volume_fp")),
        "open_interest": _num(market.get("open_interest_fp")),
        "market_open_time": _ts(market.get("open_time")),
        "tip_off": tip_off,
    }


def is_luka_pm_market(market: dict) -> bool:
    return _ascii(market.get("question", "")).startswith(_ascii(PLAYER_NAME) + ":")


def pm_snapshot(market: dict, game: Game, now: datetime, tip_off: datetime | None) -> dict | None:
    stat = PM_STATS.get(market.get("sportsMarketType"))
    if stat is None or market.get("line") is None or market.get("closed"):
        return None
    line = float(market["line"])
    strike = at_least_strike(line)
    if strike is None:
        return None
    # Polymarket quotes the Yes (over) token. No is its complement on a binary market.
    yes_bid, yes_ask = _num(market.get("bestBid")), _num(market.get("bestAsk"))
    return {
        "snapshot_at": now,
        "meta": {"venue": "polymarket", "game_key": game.key, "market_id": str(market["id"]), "stat": stat, "strike": strike},
        "line": line,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": None if yes_ask is None else round(1 - yes_ask, 4),
        "no_ask": None if yes_bid is None else round(1 - yes_bid, 4),
        "last": _num(market.get("lastTradePrice")),
        "volume": _num(market.get("volumeNum")),
        "liquidity": _num(market.get("liquidityNum")),
        "market_open_time": _ts(market.get("createdAt")),
        "tip_off": tip_off,
    }


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _get(session: requests.Session, url: str, params: dict | None = None):
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def upcoming_games(session: requests.Session) -> list[Game]:
    """Lakers games that have an open game-winner market on Kalshi."""
    events, cursor = set(), None
    while True:
        params = {"series_ticker": KALSHI_GAME_SERIES, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = _get(session, f"{KALSHI}/markets", params)
        events |= {m["event_ticker"] for m in page.get("markets", [])}
        cursor = page.get("cursor")
        if not cursor:
            break
    games = {g for g in map(parse_kalshi_event, events) if g and TEAM in (g.away, g.home)}
    return sorted(games, key=lambda g: g.game_date)


def snapshot_game(
    session: requests.Session, game: Game, now: datetime
) -> tuple[datetime | None, list[dict], list[dict], list[str]]:
    """Returns (tip_off, snapshots, raw cache docs, errors) for one game.

    A venue failing doesn't stop the other.
    """
    snapshots, raw, errors = [], [], []

    tip_off, pm_markets = None, []
    try:
        events = _get(session, f"{GAMMA}/events", {"slug": game.polymarket_slug})
        if events:
            tip_off = _ts(events[0].get("startTime"))
            pm_markets = [m for m in events[0].get("markets", []) if is_luka_pm_market(m)]
    except requests.RequestException as e:
        errors.append(f"polymarket: {e}")

    if tip_off and now >= tip_off:
        return tip_off, [], [], errors  # in-game prices are useless for pre-game decisions

    for m in pm_markets:
        if doc := pm_snapshot(m, game, now, tip_off):
            snapshots.append(doc)
    if pm_markets:
        raw.append({"source": "polymarket", "key": game.polymarket_slug, "fetched_at": now,
                    "payload": [_trim(m, PM_RAW_FIELDS) for m in pm_markets]})

    for stat, series in KALSHI_SERIES.items():
        event_ticker = f"{series}-{game.kalshi_suffix}"
        try:
            page = _get(session, f"{KALSHI}/markets", {"event_ticker": event_ticker, "status": "open", "limit": 200})
        except requests.RequestException as e:
            errors.append(f"kalshi {event_ticker}: {e}")
            continue
        luka = [m for m in page.get("markets", []) if KALSHI_PLAYER_TAG in m["ticker"]]
        for m in luka:
            if doc := kalshi_snapshot(m, stat, game, now, tip_off):
                snapshots.append(doc)
        if luka:
            raw.append({"source": "kalshi", "key": event_ticker, "fetched_at": now,
                        "payload": [_trim(m, KALSHI_RAW_FIELDS) for m in luka]})

    return tip_off, snapshots, raw, errors


def run(db, session: requests.Session, now: datetime | None = None, dry_run: bool = False) -> bool:
    """Snapshot every upcoming Lakers game. Returns False if any request failed."""
    now = now or datetime.now(timezone.utc)
    ok = True
    for game in upcoming_games(session):
        tip_off, snapshots, raw, errors = snapshot_game(session, game, now)
        venues = {v: sum(s["meta"]["venue"] == v for s in snapshots) for v in ("kalshi", "polymarket")}
        tip = f"tip {tip_off:%Y-%m-%d %H:%MZ}" if tip_off else "tip unknown"
        if tip_off and now >= tip_off:
            tip += " (started, skipped)"
        print(f"{game.key}  {tip}  kalshi {venues['kalshi']}  polymarket {venues['polymarket']}")
        for e in errors:
            print(f"  ERROR {e}")
        ok &= not errors
        if dry_run:
            continue
        if snapshots:
            db.market_snapshots.insert_many(snapshots)
        if raw:
            db.raw_api_cache.insert_many(raw)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="fetch and print, but write nothing")
    args = parser.parse_args()

    from luka.db.client import get_db

    db = None if args.dry_run else get_db()
    sys.exit(0 if run(db, make_session(), dry_run=args.dry_run) else 1)


if __name__ == "__main__":
    main()
