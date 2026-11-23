import os
from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv

from luka.db.schema import ensure_schema
from luka.ingest.teams import possessions, run, team_game_docs, team_player_doc

load_dotenv()

# Shapes copied from real TeamGameLogs / PlayerGameLogs rows (trimmed).
LAL = {"SEASON_YEAR": "2026-27", "TEAM_ID": 1610612747, "GAME_ID": "0022600268", "GAME_DATE": "2026-11-20T00:00:00",
       "MATCHUP": "LAL vs. SAC", "MIN": 240.0, "PTS": 118, "FGA": 88, "FTA": 25, "OREB": 10, "TOV": 12}
SAC = {"SEASON_YEAR": "2026-27", "TEAM_ID": 1610612758, "GAME_ID": "0022600268", "GAME_DATE": "2026-11-20T00:00:00",
       "MATCHUP": "SAC @ LAL", "MIN": 240.0, "PTS": 104, "FGA": 90, "FTA": 18, "OREB": 12, "TOV": 15}
PLAYER = {"SEASON_YEAR": "2026-27", "PLAYER_ID": 1630559, "PLAYER_NAME": "Austin Reaves", "GAME_ID": "0022600268",
          "GAME_DATE": "2026-11-20T00:00:00", "MATCHUP": "LAL vs. SAC", "MIN": 33.5, "PTS": 21, "REB": 4, "AST": 6,
          "FGA": 15, "FTA": 6, "TOV": 2}


def test_possessions_formula():
    assert possessions(LAL) == pytest.approx(88 + 0.44 * 25 - 10 + 12)


def test_team_game_docs_pairs_both_sides():
    docs = {d["team"]: d for d in team_game_docs([LAL, SAC])}
    assert docs["LAL"]["opp_pts"] == 104 and docs["SAC"]["opp_pts"] == 118
    assert docs["LAL"]["home"] is True and docs["SAC"]["home"] is False
    assert docs["LAL"]["opponent"] == "SAC"
    assert docs["LAL"]["poss"] == docs["SAC"]["poss"] == round((101.0 + 100.92) / 2, 1)
    assert docs["LAL"]["game_date"] == datetime(2026, 11, 20, tzinfo=timezone.utc)


def test_team_game_docs_skips_unpaired_and_preseason_games():
    assert team_game_docs([LAL]) == []
    assert team_game_docs([{**LAL, "GAME_ID": "0012600001"}, {**SAC, "GAME_ID": "0012600001"}]) == []


def test_team_player_doc():
    doc = team_player_doc(PLAYER)
    assert (doc["player_id"], doc["name"], doc["team"]) == (1630559, "Austin Reaves", "LAL")
    assert (doc["minutes"], doc["fga"], doc["fta"], doc["tov"]) == (33.5, 15, 6, 2)


def test_run_keeps_going_when_one_request_fails():
    def league(season, season_type):
        raise RuntimeError("blocked")

    def players(team, season, season_type):
        return [PLAYER] if season_type == "Regular Season" else []

    assert run(None, ["2026-27"], teams_for=lambda s: ["LAL"], fetch_league=league,
               fetch_team_players=players, dry_run=True, delay_s=0) is False


atlas = pytest.mark.skipif(not os.environ.get("MONGODB_URI"), reason="MONGODB_URI not set")


@pytest.fixture(scope="module")
def db():
    from luka.db.client import get_client

    name = "luka_test"
    get_client().drop_database(name)
    test_db = get_client()[name]
    ensure_schema(test_db)
    yield test_db
    get_client().drop_database(name)


@atlas
def test_run_upserts_and_passes_validators(db):
    def league(season, season_type):
        return [LAL, SAC] if season_type == "Regular Season" else []

    def players(team, season, season_type):
        return [PLAYER] if season_type == "Regular Season" else []

    for _ in range(2):  # second run must not duplicate
        assert run(db, ["2026-27"], teams_for=lambda s: ["LAL"], fetch_league=league,
                   fetch_team_players=players, delay_s=0)
    assert db.team_games.count_documents({}) == 2
    assert db.team_player_games.count_documents({}) == 1
