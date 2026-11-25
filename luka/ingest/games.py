"""Load Luka's career game logs from stats.nba.com (via nba_api) into the games collection.

stats.nba.com returns one season and one season type per request, so a full career load is
about 27 requests (9 seasons x regular season, playoffs and play-in). Preseason is never
requested. Each game is upserted by game_id, so re-running updates rows and never duplicates
them. Run it after every game day to pick up new games.

    python -m luka.ingest.games                      # whole career
    python -m luka.ingest.games --current            # this season only, plus team data and schedule (the daily update)
    python -m luka.ingest.games --season 2024-25     # one season; repeatable
    python -m luka.ingest.games --dry-run            # fetch and print, write nothing
"""
import argparse
import sys
import time
from datetime import date, datetime, timezone
from typing import Callable

import requests
from pymongo import UpdateOne

from luka.config import PLAYER_ID, PLAYER_NAME, ROOKIE_SEASON

# The request parameter, and the label we store. Play-in games have their own type.
SEASON_TYPES = {"Regular Season": "regular", "Playoffs": "playoffs", "PlayIn": "play_in"}

# Every NBA game id starts with its type: 002 regular, 004 playoffs, 005 play-in, 006 Cup final.
# We label by prefix, not by which request returned the row, so a mislabelled response can't
# leak a different kind of game into a season.
SEASON_TYPE_BY_PREFIX = {"002": "regular", "004": "playoffs", "005": "play_in", "006": "cup_final"}

REQUEST_DELAY_S = 0.8  # pause between requests; stats.nba.com throttles fast clients
RETRIES = 3
TIMEOUT_S = 30

# Rows have 72 columns, half of them league ranks we never use. Dropping *_RANK halves the cache.
def _trim_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.endswith("_RANK")}


def season_label(start_year: int) -> str:
    """2026 -> '2026-27'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season_start(today: date) -> int:
    """NBA seasons start in October, so Sep 2027 is still the 2026-27 season."""
    return today.year if today.month >= 10 else today.year - 1


def career_seasons(today: date) -> list[str]:
    return [season_label(y) for y in range(ROOKIE_SEASON, current_season_start(today) + 1)]


def parse_matchup(matchup: str) -> tuple[str, str, bool]:
    """'LAL vs. HOU' -> (LAL, HOU, home=True); 'LAL @ DAL' -> (LAL, DAL, home=False)."""
    if " vs. " in matchup:
        team, opp = matchup.split(" vs. ")
        return team.strip(), opp.strip(), True
    if " @ " in matchup:
        team, opp = matchup.split(" @ ")
        return team.strip(), opp.strip(), False
    raise ValueError(f"unrecognised MATCHUP {matchup!r}")


def game_doc(row: dict) -> dict | None:
    """One stats.nba.com game-log row -> one games document. None for game types we don't keep."""
    game_id = row["GAME_ID"]
    season_type = SEASON_TYPE_BY_PREFIX.get(game_id[:3])
    if season_type is None:
        return None
    team, opponent, home = parse_matchup(row["MATCHUP"])
    # GAME_DATE is the US Eastern calendar date ("2026-11-15T00:00:00"), the same date Kalshi and
    # Polymarket put in their tickers. We store it as midnight UTC on that date.
    day = date.fromisoformat(row["GAME_DATE"][:10])
    away, home_team = (opponent, team) if home else (team, opponent)
    return {
        "game_id": game_id,
        "game_key": f"{day.isoformat()}_{away}_{home_team}",  # joins to market_snapshots.meta.game_key
        "player_id": int(row["PLAYER_ID"]),
        "season": row["SEASON_YEAR"],
        "season_type": season_type,
        "game_date": datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
        "team": team,
        "opponent": opponent,
        "home": home,
        "minutes": round(float(row["MIN"] or 0), 2),
        "pts": int(row["PTS"] or 0),
        "reb": int(row["REB"] or 0),
        "ast": int(row["AST"] or 0),
    }


def fetch_rows(season: str, season_type: str) -> list[dict]:
    """One season of Luka's game logs from stats.nba.com."""
    from nba_api.stats.endpoints import playergamelogs  # slow import; keep it out of tests

    return call_stats_api(
        playergamelogs.PlayerGameLogs, "PlayerGameLogs", f"{season} {season_type}",
        player_id_nullable=PLAYER_ID, season_nullable=season, season_type_nullable=season_type,
    )


def call_stats_api(endpoint, result_key: str, label: str, **params) -> list[dict]:
    """Call one nba_api endpoint with retries and a timeout; return its rows as dicts.

    stats.nba.com often stalls or returns an HTML block page instead of JSON (common from cloud
    IPs). Both surface as exceptions here, so we retry with growing pauses, then raise
    RuntimeError so the caller can log it and move on to the next request.
    """
    for attempt in range(1, RETRIES + 1):
        try:
            return endpoint(timeout=TIMEOUT_S, **params).get_normalized_dict()[result_key]
        except (requests.RequestException, ValueError, KeyError) as e:  # ValueError covers bad JSON
            if attempt == RETRIES:
                raise RuntimeError(f"{label}: {type(e).__name__}: {e}") from e
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


Fetch = Callable[[str, str], list[dict]]


def run(db, seasons: list[str], fetch: Fetch = fetch_rows, dry_run: bool = False, delay_s: float = REQUEST_DELAY_S) -> bool:
    """Fetch and upsert each season. Returns False if any request failed (others still run)."""
    now = datetime.now(timezone.utc)
    ok, total = True, 0
    if not dry_run:
        db.players.update_one(
            {"player_id": PLAYER_ID}, {"$set": {"player_id": PLAYER_ID, "name": PLAYER_NAME}}, upsert=True
        )
    for season in seasons:
        for season_type, label in SEASON_TYPES.items():
            try:
                rows = fetch(season, season_type)
            except RuntimeError as e:
                print(f"  ERROR {e}")
                ok = False
                continue
            finally:
                time.sleep(delay_s)
            docs = [d for d in map(game_doc, rows) if d]
            if rows or label == "regular":
                print(f"{season} {label:<8} {len(docs):>3} games")
            total += len(docs)
            if dry_run or not rows:
                continue
            # Cache the raw rows first (CLAUDE.md: cache before normalizing), then upsert.
            db.raw_api_cache.insert_one({
                "source": "nba_api", "key": f"playergamelogs/{PLAYER_ID}/{season}/{season_type}",
                "fetched_at": now, "payload": [_trim_row(r) for r in rows],
            })
            if docs:
                db.games.bulk_write(
                    [UpdateOne({"game_id": d["game_id"]}, {"$set": {**d, "updated_at": now}}, upsert=True) for d in docs],
                    ordered=False,
                )
    print(f"total {total} games" + (" (dry run, nothing written)" if dry_run else ""))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", action="append", help="e.g. 2026-27; repeatable. Default: whole career")
    parser.add_argument("--current", action="store_true", help="only the current season (the daily update)")
    parser.add_argument("--dry-run", action="store_true", help="fetch and print, but write nothing")
    args = parser.parse_args()

    from luka.db.client import get_db

    today = datetime.now(timezone.utc).date()
    if args.current:
        seasons = [season_label(current_season_start(today))]
    else:
        seasons = args.season or career_seasons(today)
    db = None if args.dry_run else get_db()
    ok = run(db, seasons, dry_run=args.dry_run)
    if args.current and not args.dry_run:
        # The daily scheduled task runs only this module, so it refreshes team data and the
        # schedule (tip-off times for live predictions) too, then scores finished predictions.
        from luka.ingest import schedule, teams
        from luka.model import score

        ok &= teams.run(db, seasons)
        ok &= schedule.run(db, seasons[0])
        ok &= score.run(db)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
