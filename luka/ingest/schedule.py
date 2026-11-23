"""Load the season schedule for Luka's team into the schedule collection.

The games collection only holds finished games, so this is where an upcoming game gets its
stats.nba.com game_id and tip-off time (docs/AGENTS.md, risk 3). The prediction step uses it
to decide when to predict, and the market snapshots use it to skip quiet days.

One request (stats.nba.com scheduleleaguev2) returns the whole league's season. We keep
regular season, play-in, playoff and Cup final games for TEAM and upsert them by game_id,
so tip-off changes (flexed TV games, postponements) overwrite the old time on the next run.
Runs daily inside `luka.ingest.games --current`.

    python -m luka.ingest.schedule              # this season
    python -m luka.ingest.schedule --dry-run    # fetch and print, write nothing
"""
import argparse
import sys
from datetime import date, datetime, timezone
from typing import Callable

from pymongo import UpdateOne

from luka.config import TEAM
from luka.ingest.games import SEASON_TYPE_BY_PREFIX, call_stats_api, current_season_start, season_label

RAW_FIELDS = [
    "gameId", "seasonYear", "gameDateEst", "gameDateTimeUTC", "gameStatus", "gameStatusText", "postponedStatus",
    "ifNecessary", "gameLabel", "homeTeam_teamTricode", "awayTeam_teamTricode",
]
STATUS = {1: "scheduled", 2: "live", 3: "final"}


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def schedule_doc(row: dict, team: str = TEAM) -> dict | None:
    """One scheduleleaguev2 row -> one schedule doc, from `team`'s side. None if not kept."""
    game_id = row["gameId"]
    season_type = SEASON_TYPE_BY_PREFIX.get(game_id[:3])  # drops preseason (001) and All-Star (003)
    away, home = row["awayTeam_teamTricode"], row["homeTeam_teamTricode"]
    if season_type is None or team not in (away, home):
        return None
    # A playoff "if necessary" game may never be played; predict it only once it's confirmed.
    if row.get("ifNecessary") == "true":
        return None
    # gameDateEst is the US Eastern date ("2026-11-23T00:00:00Z"); like games.game_date we store
    # it as midnight UTC on that date, so game_key matches games and market_snapshots.
    day = date.fromisoformat(row["gameDateEst"][:10])
    status = "postponed" if row.get("postponedStatus", "N") != "N" else STATUS.get(row["gameStatus"], "scheduled")
    return {
        "game_id": game_id,
        "game_key": f"{day.isoformat()}_{away}_{home}",
        "season": row["seasonYear"],
        "season_type": season_type,
        "game_date": datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
        "tip_off": _utc(row["gameDateTimeUTC"]),
        # Unscheduled TV slots show "TBD" instead of a time; the tip_off above is then a placeholder.
        "tip_time_tbd": "TBD" in (row.get("gameStatusText") or "").upper(),
        "team": team,
        "opponent": away if home == team else home,
        "home": home == team,
        "status": status,
    }


def fetch_rows(season: str) -> list[dict]:
    from nba_api.stats.endpoints import scheduleleaguev2  # slow import; keep it out of tests

    return call_stats_api(
        scheduleleaguev2.ScheduleLeagueV2, "SeasonGames", f"schedule {season}", season=season, league_id="00",
    )


def run(db, season: str, fetch: Callable[[str], list[dict]] = fetch_rows, dry_run: bool = False) -> bool:
    """Fetch and upsert one season. Returns False if the request failed."""
    now = datetime.now(timezone.utc)
    try:
        rows = fetch(season)
    except RuntimeError as e:
        print(f"  ERROR {e}")
        return False
    mine = [r for r in rows if TEAM in (r.get("homeTeam_teamTricode"), r.get("awayTeam_teamTricode"))]
    docs = [d for d in map(schedule_doc, mine) if d]
    upcoming = [d for d in docs if d["status"] == "scheduled"]
    print(f"schedule {season}: {len(docs)} {TEAM} games, {len(upcoming)} still to play")
    if upcoming:
        nxt = min(upcoming, key=lambda d: d["tip_off"])
        print(f"  next: {nxt['game_key']} ({nxt['game_id']}) tip {nxt['tip_off']:%Y-%m-%d %H:%MZ}")
    if dry_run or not docs:
        return True
    # Cache the raw rows first (CLAUDE.md), trimmed: a full row has ~100 columns of leaders and broadcasters.
    db.raw_api_cache.insert_one({
        "source": "nba_api", "key": f"scheduleleaguev2/{season}/{TEAM}", "fetched_at": now,
        "payload": [{k: r.get(k) for k in RAW_FIELDS} for r in mine],
    })
    db.schedule.bulk_write(
        [UpdateOne({"game_id": d["game_id"]}, {"$set": {**d, "updated_at": now}}, upsert=True) for d in docs],
        ordered=False,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", help="e.g. 2026-27 (default: current season)")
    parser.add_argument("--dry-run", action="store_true", help="fetch and print, but write nothing")
    args = parser.parse_args()

    from luka.db.client import get_db

    season = args.season or season_label(current_season_start(datetime.now(timezone.utc).date()))
    db = None if args.dry_run else get_db()
    sys.exit(0 if run(db, season, dry_run=args.dry_run) else 1)


if __name__ == "__main__":
    main()
