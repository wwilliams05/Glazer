from datetime import datetime, timezone

import pytest

from luka.ingest.schedule import run, schedule_doc


def _row(**changes) -> dict:
    """Shape copied from a real scheduleleaguev2 SeasonGames row (trimmed)."""
    row = {
        "gameId": "0022600295", "seasonYear": "2026-27", "gameDateEst": "2026-11-23T00:00:00Z",
        "gameDateTimeUTC": "2026-11-24T03:00:00Z", "gameStatus": 1, "gameStatusText": "10:00 pm ET",
        "postponedStatus": "N", "ifNecessary": "false", "gameLabel": "",
        "homeTeam_teamTricode": "LAL", "awayTeam_teamTricode": "UTA",
    }
    row.update(changes)
    return row


def test_home_game():
    d = schedule_doc(_row())
    assert d["game_id"] == "0022600295"
    assert d["game_key"] == "2026-11-23_UTA_LAL"  # Eastern date, away then home, like markets
    assert d["game_date"] == datetime(2026, 11, 23, tzinfo=timezone.utc)
    assert d["tip_off"] == datetime(2026, 11, 24, 3, 0, tzinfo=timezone.utc)  # 7pm Pacific is next day UTC
    assert (d["team"], d["opponent"], d["home"]) == ("LAL", "UTA", True)
    assert d["status"] == "scheduled" and d["season_type"] == "regular" and not d["tip_time_tbd"]


def test_away_cup_game():
    d = schedule_doc(_row(gameId="0022600071", homeTeam_teamTricode="SAS", awayTeam_teamTricode="LAL",
                          gameLabel="Emirates NBA Cup"))
    assert (d["team"], d["opponent"], d["home"]) == ("LAL", "SAS", False)
    assert d["season_type"] == "regular"  # Cup group games count as regular season


@pytest.mark.parametrize("changes", [
    {"gameId": "0012600009"},  # preseason
    {"homeTeam_teamTricode": "BOS", "awayTeam_teamTricode": "NYK"},  # not our team
    {"gameId": "0042600101", "ifNecessary": "true"},  # playoff game that may never happen
], ids=["preseason", "other-teams", "if-necessary"])
def test_games_we_skip(changes):
    assert schedule_doc(_row(**changes)) is None


def test_status_mapping():
    assert schedule_doc(_row(gameStatus=3, gameStatusText="Final"))["status"] == "final"
    assert schedule_doc(_row(gameStatus=2))["status"] == "live"
    assert schedule_doc(_row(postponedStatus="Y"))["status"] == "postponed"
    assert schedule_doc(_row(gameStatusText="TBD"))["tip_time_tbd"]


def test_dry_run_writes_nothing():
    assert run(None, "2026-27", fetch=lambda season: [_row(), _row(gameId="0012600009")], dry_run=True)


def test_failed_request_returns_false():
    def fail(season):
        raise RuntimeError("blocked")
    assert not run(None, "2026-27", fetch=fail)
