import os
from datetime import date, datetime, timezone

import pytest
from dotenv import load_dotenv

from luka.db.schema import ensure_schema
from luka.ingest.games import career_seasons, current_season_start, game_doc, parse_matchup, run, season_label

load_dotenv()

# Shape copied from a real PlayerGameLogs row (trimmed).
ROW = {
    "SEASON_YEAR": "2026-27",
    "PLAYER_ID": 1629029,
    "TEAM_ABBREVIATION": "LAL",
    "GAME_ID": "0022600237",
    "GAME_DATE": "2026-11-15T00:00:00",
    "MATCHUP": "LAL @ PHX",
    "MIN": 35.733333,
    "PTS": 43,
    "REB": 3,
    "AST": 9,
    "PTS_RANK": 1,
}


def test_season_helpers():
    assert season_label(2018) == "2018-19"
    assert season_label(2099) == "2099-00"
    assert current_season_start(date(2026, 11, 16)) == 2026
    assert current_season_start(date(2027, 6, 1)) == 2026  # Finals belong to the season that started in 2026
    assert career_seasons(date(2026, 11, 16))[0] == "2018-19"
    assert career_seasons(date(2026, 11, 16))[-1] == "2026-27"


@pytest.mark.parametrize(
    "matchup,expected",
    [("LAL vs. HOU", ("LAL", "HOU", True)), ("LAL @ DAL", ("LAL", "DAL", False)), ("DAL vs. LAL", ("DAL", "LAL", True))],
)
def test_parse_matchup(matchup, expected):
    assert parse_matchup(matchup) == expected


def test_parse_matchup_rejects_unknown_format():
    with pytest.raises(ValueError):
        parse_matchup("LAL - PHX")


def test_game_doc_away_game():
    doc = game_doc(ROW)
    assert doc["game_key"] == "2026-11-15_LAL_PHX"  # away first, same as market_snapshots
    assert (doc["team"], doc["opponent"], doc["home"]) == ("LAL", "PHX", False)
    assert doc["game_date"] == datetime(2026, 11, 15, tzinfo=timezone.utc)
    assert (doc["minutes"], doc["pts"], doc["reb"], doc["ast"]) == (35.73, 43, 3, 9)
    assert doc["season_type"] == "regular"


def test_game_doc_home_game_key_puts_opponent_first():
    doc = game_doc({**ROW, "MATCHUP": "LAL vs. CHA"})
    assert doc["game_key"] == "2026-11-15_CHA_LAL"


@pytest.mark.parametrize("prefix,label", [("004", "playoffs"), ("005", "play_in"), ("006", "cup_final"), ("001", None)])
def test_game_doc_labels_by_game_id_prefix(prefix, label):
    doc = game_doc({**ROW, "GAME_ID": prefix + "2600237"})
    assert (doc and doc["season_type"]) == label  # 001 = preseason, dropped


def test_run_keeps_going_when_one_request_fails():
    def fetch(season, season_type):
        if season_type == "Playoffs":
            raise RuntimeError("blocked")
        return [ROW] if season_type == "Regular Season" else []

    assert run(None, ["2026-27"], fetch=fetch, dry_run=True, delay_s=0) is False


# ---- Against a real cluster: upserts must be idempotent and pass the validator ----

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
def test_run_upserts_without_duplicates(db):
    rows = [ROW, {**ROW, "GAME_ID": "0022600222", "GAME_DATE": "2026-11-11T00:00:00", "MATCHUP": "LAL @ MIL"}]

    def fetch(season, season_type):
        return rows if season_type == "Regular Season" else []

    assert run(db, ["2026-27"], fetch=fetch, delay_s=0)
    rows[0] = {**ROW, "PTS": 44}  # a stat correction arrives on the next run
    assert run(db, ["2026-27"], fetch=fetch, delay_s=0)

    assert db.games.count_documents({}) == 2
    assert db.games.find_one({"game_id": "0022600237"})["pts"] == 44
    assert db.players.count_documents({}) == 1
    assert db.raw_api_cache.count_documents({"source": "nba_api"}) == 2
