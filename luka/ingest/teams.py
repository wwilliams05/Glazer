"""Load team-level game data from stats.nba.com into team_games and team_player_games.

- team_games: every team's box score for every game, league-wide (one request per season and
  season type). Gives each opponent's points allowed and possessions, for defense and pace.
- team_player_games: every player's box score for Luka's team(s) in each season (one request
  per team, season and season type). Gives who played and how much, for "teammates out".

Luka's team per season is read from the games collection, so load games first. In 2024-25
that is both DAL and LAL; only games Luka's team played matter, and the game_id join in the
feature pipeline takes care of the trade date.

Upserts on (game_id, team_id) and (game_id, player_id), so re-running is safe.

    python -m luka.ingest.teams                      # whole career
    python -m luka.ingest.teams --season 2026-27     # one season; repeatable
    python -m luka.ingest.teams --dry-run
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Callable

from pymongo import UpdateOne

from luka.ingest.games import (
    REQUEST_DELAY_S, SEASON_TYPE_BY_PREFIX, SEASON_TYPES, _trim_row, call_stats_api, career_seasons,
    current_season_start, game_doc, parse_matchup, season_label,
)


def _day(row: dict) -> datetime:
    return datetime.fromisoformat(row["GAME_DATE"][:10]).replace(tzinfo=timezone.utc)


def possessions(row: dict) -> float:
    """Standard box-score estimate: FGA + 0.44*FTA - OREB + TOV."""
    return row["FGA"] + 0.44 * row["FTA"] - row["OREB"] + row["TOV"]


def team_game_docs(rows: list[dict]) -> list[dict]:
    """Pair the two team rows of each game into one doc per team with opponent points.

    Possessions are averaged over both teams, the usual way to cancel out each side's
    estimation error. A game with only one row (bad data) is skipped.
    """
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["GAME_ID"], []).append(r)
    docs = []
    for game_id, pair in by_game.items():
        season_type = SEASON_TYPE_BY_PREFIX.get(game_id[:3])
        if season_type is None or len(pair) != 2:
            continue
        poss = round((possessions(pair[0]) + possessions(pair[1])) / 2, 1)
        for me, opp in (pair, pair[::-1]):
            team, opponent, home = parse_matchup(me["MATCHUP"])
            docs.append({
                "game_id": game_id,
                "team_id": int(me["TEAM_ID"]),
                "team": team,
                "season": me["SEASON_YEAR"],
                "season_type": season_type,
                "game_date": _day(me),
                "opponent": opponent,
                "home": home,
                "minutes": round(float(me["MIN"] or 0)),
                "pts": int(me["PTS"]),
                "opp_pts": int(opp["PTS"]),
                "poss": poss,
            })
    return docs


def team_player_doc(row: dict) -> dict | None:
    game = game_doc(row)  # reuses the game-type and matchup handling
    if game is None:
        return None
    return {
        "game_id": game["game_id"],
        "player_id": int(row["PLAYER_ID"]),
        "name": row["PLAYER_NAME"],
        "team": game["team"],
        "season": game["season"],
        "season_type": game["season_type"],
        "game_date": game["game_date"],
        "minutes": game["minutes"],
        "pts": game["pts"],
        "reb": game["reb"],
        "ast": game["ast"],
        "fga": int(row["FGA"] or 0),
        "fta": int(row["FTA"] or 0),
        "tov": int(row["TOV"] or 0),
    }


def fetch_league(season: str, season_type: str) -> list[dict]:
    from nba_api.stats.endpoints import teamgamelogs

    return call_stats_api(
        teamgamelogs.TeamGameLogs, "TeamGameLogs", f"league {season} {season_type}",
        season_nullable=season, season_type_nullable=season_type,
    )


def fetch_team_players(team: str, season: str, season_type: str) -> list[dict]:
    from nba_api.stats.endpoints import playergamelogs
    from nba_api.stats.static import teams

    team_id = teams.find_team_by_abbreviation(team)["id"]
    return call_stats_api(
        playergamelogs.PlayerGameLogs, "PlayerGameLogs", f"{team} {season} {season_type}",
        team_id_nullable=team_id, season_nullable=season, season_type_nullable=season_type,
    )


def lukas_teams(db, season: str) -> list[str]:
    return sorted(db.games.distinct("team", {"season": season}))


def _upsert(db, collection: str, docs: list[dict], keys: tuple[str, ...], now: datetime) -> None:
    if docs:
        db[collection].bulk_write(
            [UpdateOne({k: d[k] for k in keys}, {"$set": {**d, "updated_at": now}}, upsert=True) for d in docs],
            ordered=False,
        )


def run(
    db,
    seasons: list[str],
    teams_for: Callable[[str], list[str]] | None = None,
    fetch_league: Callable[[str, str], list[dict]] = fetch_league,
    fetch_team_players: Callable[[str, str, str], list[dict]] = fetch_team_players,
    dry_run: bool = False,
    delay_s: float = REQUEST_DELAY_S,
) -> bool:
    """Fetch and upsert each season. Returns False if any request failed (others still run)."""
    now = datetime.now(timezone.utc)
    teams_for = teams_for or (lambda season: lukas_teams(db, season))
    ok = True

    def fetch(fn, *args) -> list[dict] | None:
        nonlocal ok
        try:
            return fn(*args)
        except RuntimeError as e:
            print(f"  ERROR {e}")
            ok = False
            return None
        finally:
            time.sleep(delay_s)

    for season in seasons:
        team_list = teams_for(season)
        for season_type, label in SEASON_TYPES.items():
            rows = fetch(fetch_league, season, season_type)
            if rows:
                docs = team_game_docs(rows)
                print(f"{season} {label:<8} league  {len(docs) // 2:>4} games")
                if not dry_run:
                    db.raw_api_cache.insert_one({"source": "nba_api", "key": f"teamgamelogs/{season}/{season_type}",
                                                 "fetched_at": now, "payload": [_trim_row(r) for r in rows]})
                    _upsert(db, "team_games", docs, ("game_id", "team_id"), now)

            # A team's playoff request is wasted if Luka has no games of that type that season.
            if label != "regular" and db is not None and not db.games.count_documents({"season": season, "season_type": label}):
                continue
            for team in team_list:
                rows = fetch(fetch_team_players, team, season, season_type)
                if not rows:
                    continue
                docs = [d for d in map(team_player_doc, rows) if d]
                print(f"{season} {label:<8} {team:<5}  {len(docs):>4} player-games")
                if not dry_run:
                    db.raw_api_cache.insert_one({"source": "nba_api", "key": f"playergamelogs/team/{team}/{season}/{season_type}",
                                                 "fetched_at": now, "payload": [_trim_row(r) for r in rows]})
                    _upsert(db, "team_player_games", docs, ("game_id", "player_id"), now)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", action="append", help="e.g. 2026-27; repeatable. Default: whole career")
    parser.add_argument("--current", action="store_true", help="only the current season")
    parser.add_argument("--dry-run", action="store_true", help="fetch and print, but write nothing")
    args = parser.parse_args()

    from luka.db.client import get_db

    today = datetime.now(timezone.utc).date()
    seasons = [season_label(current_season_start(today))] if args.current else (args.season or career_seasons(today))
    db = get_db()  # read even on a dry run: Luka's teams come from the games collection
    sys.exit(0 if run(db, seasons, dry_run=args.dry_run) else 1)


if __name__ == "__main__":
    main()
