import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from dotenv import load_dotenv

from luka.db.schema import ensure_schema
from luka.features.pipeline import (
    FEATURE_COLUMNS, add_form_fallbacks, build, feature_docs, opponent_features, teammate_features, write,
)

load_dotenv()

LUKA = 1629029
T0 = datetime(2025, 10, 22, tzinfo=timezone.utc)
day = lambda n: T0 + timedelta(days=n)  # noqa: E731


# ---- pandas parts (no database) ---------------------------------------------------------

def _team_players(games: dict[str, tuple[int, list[tuple[int, float]]]]) -> pd.DataFrame:
    """games: game_id -> (day, [(player_id, minutes), ...]) for team AAA."""
    rows = []
    for gid, (d, players) in games.items():
        for pid, mins in players:
            rows.append({"game_id": gid, "player_id": pid, "team": "AAA", "game_date": day(d),
                         "minutes": mins, "fga": 10, "fta": 0, "tov": 0})
    return pd.DataFrame(rows)


def test_teammate_out_detected_from_earlier_games_only():
    regulars = [(LUKA, 36), (1, 34), (2, 30), (3, 8)]  # player 3 is not rotation (8 min)
    games = {f"G{i}": (i, regulars) for i in range(10)}
    games["G10"] = (10, [(LUKA, 36), (2, 30), (3, 8)])  # player 1 sits
    tp = _team_players(games)
    luka = pd.DataFrame([{"game_id": "G10", "game_date": day(10), "team": "AAA"}])
    out = teammate_features(luka, tp, LUKA).iloc[0]
    assert out["n_rotation_out"] == 1
    assert out["top_teammate_out"] == 1  # 1 and 2 tie on load; idxmax takes the first, player 1
    # team load per game is 40 (4 players x 10 FGA); player 1 averaged 10 of it
    assert out["usage_share_out"] == pytest.approx(10 / 40)


def test_teammates_unknown_without_enough_history():
    games = {f"G{i}": (i, [(LUKA, 36), (1, 34)]) for i in range(3)}
    luka = pd.DataFrame([{"game_id": "G2", "game_date": day(2), "team": "AAA"}])
    assert pd.isna(teammate_features(luka, _team_players(games), LUKA).iloc[0]["n_rotation_out"])


def test_luka_himself_is_never_counted_out():
    games = {f"G{i}": (i, [(LUKA, 36), (1, 34)]) for i in range(10)}
    games["G10"] = (10, [(1, 34)])  # a game Luka missed: not one of his rows, but in the team schedule
    games["G11"] = (11, [(LUKA, 36), (1, 34)])
    luka = pd.DataFrame([{"game_id": "G11", "game_date": day(11), "team": "AAA"}])
    assert teammate_features(luka, _team_players(games), LUKA).iloc[0]["n_rotation_out"] == 0


def test_rest_days_and_season_opener_fallback():
    games = pd.DataFrame([
        {"season": "2024-25", "pts": 30, "reb": 8, "ast": 8, "minutes": 36},
        {"season": "2024-25", "pts": 20, "reb": 8, "ast": 8, "minutes": 36},
    ])
    form = pd.DataFrame([{
        "game_id": "G1", "season": "2025-26", "game_date": day(200), "prev_game_date": day(10), "home": True,
        **{f"{s}_season_to_date": None for s in ["pts", "reb", "ast", "minutes"]},
    }])
    df = add_form_fallbacks(form, games).iloc[0]
    assert df["pts_season"] == 25  # last season's average
    assert df["rest_days"] == 5 and df["back_to_back"] == 0  # capped at 5


def test_opponent_features_need_five_games():
    rows = pd.DataFrame([
        {"game_id": "G", "team": "BBB", "w_opp_pts": 1100, "w_poss": 1000, "w_min": 480, "w_games": 10,
         "lg_pts": 10000, "lg_poss": 10000, "lg_min": 4800},
        {"game_id": "H", "team": "BBB", "w_opp_pts": 440, "w_poss": 400, "w_min": 192, "w_games": 4,
         "lg_pts": 10000, "lg_poss": 10000, "lg_min": 4800},
    ])
    f = opponent_features(rows).set_index("game_id")
    assert f.loc["G", "opp_def_rtg_rel"] == pytest.approx(110 - 100)
    assert f.loc["G", "opp_pace_rel"] == pytest.approx(0)
    assert pd.isna(f.loc["H", "opp_def_rtg_rel"])


# ---- Full pipeline against a real cluster ------------------------------------------------

atlas = pytest.mark.skipif(not os.environ.get("MONGODB_URI"), reason="MONGODB_URI not set")


def _season(n_games: int) -> dict[str, list[dict]]:
    """Luka's team AAA plays BBB every game; CCC plays DDD the same nights. Luka scores 20+i."""
    games, team_games, team_players = [], [], []
    for i in range(n_games):
        gid = f"00225{i:05d}"
        home = i % 2 == 0
        games.append({"game_id": gid, "player_id": LUKA, "season": "2025-26", "season_type": "regular",
                      "game_date": day(2 * i), "team": "AAA", "opponent": "BBB", "home": home,
                      "minutes": 36, "pts": 20 + i, "reb": 8, "ast": 9})
        for gid2, a, b in ((gid, "AAA", "BBB"), (f"00226{i:05d}", "CCC", "DDD")):
            for team, opp, pts, opp_pts in ((a, b, 110, 100 + i), (b, a, 100 + i, 110)):
                team_games.append({"game_id": gid2, "team_id": hash(team) % 1000, "team": team, "season": "2025-26",
                                   "season_type": "regular", "game_date": day(2 * i), "opponent": opp,
                                   "home": team == a, "minutes": 48, "pts": pts, "opp_pts": opp_pts, "poss": 100.0})
        for pid, mins in ((LUKA, 36), (1, 34), (2, 30)):
            if pid == 1 and i == 11:
                continue  # player 1 sits game 11 (fixed, so longer seasons keep the same history)
            team_players.append({"game_id": gid, "player_id": pid, "name": f"P{pid}", "team": "AAA",
                                 "season": "2025-26", "season_type": "regular", "game_date": day(2 * i),
                                 "minutes": mins, "pts": 10, "reb": 3, "ast": 3, "fga": 10, "fta": 2, "tov": 1})
    return {"games": games, "team_games": team_games, "team_player_games": team_players}


@pytest.fixture()
def db():
    from luka.db.client import get_client

    name = "luka_test"
    get_client().drop_database(name)
    test_db = get_client()[name]
    ensure_schema(test_db)
    yield test_db
    get_client().drop_database(name)


def _load(db, data):
    for coll in ("games", "team_games", "team_player_games"):
        db[coll].delete_many({})
        db[coll].insert_many([dict(d) for d in data[coll]])


@atlas
def test_rolling_form_excludes_the_game_itself(db):
    _load(db, _season(12))
    df = build(db).set_index("game_id")
    g = df.loc["0022500010"]  # 11th game: earlier games scored 20..29
    assert g["pts_avg5"] == pytest.approx(sum(range(25, 30)) / 5)
    assert g["pts_avg10"] == pytest.approx(sum(range(20, 30)) / 10)
    assert g["pts_season"] == pytest.approx(sum(range(20, 30)) / 10)
    assert g["rest_days"] == 2 and g["back_to_back"] == 0
    assert pd.isna(df.loc["0022500000", "pts_avg5"])  # first game has no history


@atlas
def test_future_games_never_change_earlier_features(db):
    _load(db, _season(12))
    early = build(db).set_index("game_id")
    _load(db, _season(20))
    later = build(db).set_index("game_id").loc[early.index]
    pd.testing.assert_frame_equal(early[FEATURE_COLUMNS], later[FEATURE_COLUMNS])


@atlas
def test_opponent_window_and_teammate_out_end_to_end(db):
    _load(db, _season(12))
    last = build(db).set_index("game_id").loc["0022500011"]
    # BBB allowed 110 on 100 possessions every game: 10 above a league that averages 100+i
    assert last["opp_def_rtg_rel"] > 0
    assert last["n_rotation_out"] == 1


@atlas
def test_written_docs_pass_validator_and_upsert(db):
    _load(db, _season(12))
    docs = feature_docs(build(db))
    write(db, docs)
    write(db, docs)
    assert db.features.count_documents({}) == 12
    stored = db.features.find_one({"game_id": "0022500011"})
    assert stored["availability_source"] == "box_score" and stored["values"]["n_rotation_out"] == 1
    assert stored["values"]["pts_avg5"] is not None
