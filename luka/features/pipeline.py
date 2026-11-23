"""Feature pipeline: one features doc per Luka game, built only from earlier data.

Three groups of features, each from its own source:

1. Luka's form (games). MongoDB $setWindowFields: rolling 5- and 10-game averages of PTS,
   REB, AST and minutes, season-to-date averages, and days of rest. Every window ends at the
   previous game (documents [-N, -1]), so a game never sees itself.
2. Opponent strength (team_games). $setWindowFields again: the opponent's points allowed per
   100 possessions and pace over its previous 20 games, minus the league average over the
   previous 365 days, so values from 2018 and 2026 are comparable.
3. Teammates out (team_player_games). Plain pandas, because it is set logic rather than a
   rolling number: which rotation players from the team's last 10 games are missing from
   this game, and what share of the team's shot volume they took.

as_of is midnight UTC on the game's (US Eastern) date. Groups 1 and 2 use only games from
earlier dates. Group 3 uses who actually played, a stand-in for the pre-game injury report;
availability_source records that, so live rows (from the injury report) can be told apart.

    python -m luka.features.pipeline              # rebuild every game's features
    python -m luka.features.pipeline --dry-run    # compute and print a summary only
"""
import argparse
import math
from datetime import datetime, timezone

import pandas as pd
from pymongo import UpdateOne

from luka.config import PLAYER_ID

PIPELINE_VERSION = "f1"
FORM_STATS = ["pts", "reb", "ast", "minutes"]
OPP_WINDOW = 20  # opponent games
TEAM_WINDOW = 10  # team games used to decide who is in the rotation
ROTATION_MIN_MINUTES = 20.0
ROTATION_MIN_GAMES = 3
REST_CAP = 5  # 5+ days of rest all count the same; season openers would otherwise be 150+

FEATURE_COLUMNS = (
    [f"{s}_avg{n}" for s in FORM_STATS for n in (5, 10)]
    + [f"{s}_season" for s in FORM_STATS]
    + ["rest_days", "back_to_back", "home", "opp_def_rtg_rel", "opp_pace_rel",
       "n_rotation_out", "usage_share_out", "top_teammate_out"]
)


# ---- 1. Luka's form -------------------------------------------------------------------

def form_pipeline(player_id: int = PLAYER_ID) -> list[dict]:
    rolling = {
        f"{s}_avg{n}": {"$avg": f"${s}", "window": {"documents": [-n, -1]}}
        for s in FORM_STATS for n in (5, 10)
    }
    season = {f"{s}_season_to_date": {"$avg": f"${s}", "window": {"documents": ["unbounded", -1]}} for s in FORM_STATS}
    return [
        {"$match": {"player_id": player_id}},
        {"$setWindowFields": {
            "partitionBy": "$player_id",
            "sortBy": {"game_date": 1},
            "output": {**rolling, "prev_game_date": {"$shift": {"output": "$game_date", "by": -1}}},
        }},
        {"$setWindowFields": {
            "partitionBy": {"player": "$player_id", "season": "$season"},
            "sortBy": {"game_date": 1},
            "output": season,
        }},
        {"$project": {"_id": 0, "game_id": 1, "game_date": 1, "season": 1, "team": 1, "opponent": 1, "home": 1,
                      "prev_game_date": 1, **{k: 1 for k in rolling}, **{k: 1 for k in season}}},
    ]


def add_form_fallbacks(form: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Season opener: no games yet this season, so use last season's full average (known by then).

    Also turns prev_game_date into rest days.
    """
    df = form.copy()
    seasons = sorted(set(games["season"]) | set(df["season"]))
    prev = dict(zip(seasons[1:], seasons[:-1]))
    full = games.groupby("season")[FORM_STATS].mean()
    for s in FORM_STATS:
        fallback = df["season"].map(prev).map(full[s])
        df[f"{s}_season"] = df[f"{s}_season_to_date"].fillna(fallback)
    gap = (df["game_date"] - df["prev_game_date"]).dt.days
    df["rest_days"] = gap.clip(upper=REST_CAP)
    df["back_to_back"] = (gap == 1).astype(float).where(gap.notna())
    df["home"] = df["home"].astype(float)
    return df


# ---- 2. Opponent strength -------------------------------------------------------------

def opponent_pipeline(game_ids: list[str]) -> list[dict]:
    prev = {"documents": [-OPP_WINDOW, -1]}
    year = {"range": [-365, -1], "unit": "day"}  # earlier calendar days only, never the same day
    return [
        {"$setWindowFields": {
            "partitionBy": "$team",
            "sortBy": {"game_date": 1},
            "output": {
                "w_opp_pts": {"$sum": "$opp_pts", "window": prev},
                "w_poss": {"$sum": "$poss", "window": prev},
                "w_min": {"$sum": "$minutes", "window": prev},
                "w_games": {"$count": {}, "window": prev},
            },
        }},
        {"$setWindowFields": {
            "sortBy": {"game_date": 1},
            "output": {
                "lg_pts": {"$sum": "$pts", "window": year},
                "lg_poss": {"$sum": "$poss", "window": year},
                "lg_min": {"$sum": "$minutes", "window": year},
            },
        }},
        # Filter only after the windows: a $match first would drop the history they need.
        {"$match": {"game_id": {"$in": game_ids}}},
        {"$project": {"_id": 0, "game_id": 1, "team": 1, "w_opp_pts": 1, "w_poss": 1, "w_min": 1, "w_games": 1,
                      "lg_pts": 1, "lg_poss": 1, "lg_min": 1}},
    ]


def opponent_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Defensive rating = points allowed per 100 possessions; pace = possessions per 48 minutes.

    Both relative to the league over the previous year. Fewer than 5 earlier games -> missing.
    """
    df = rows.copy()
    enough = df["w_games"] >= 5
    def_rtg = 100 * df["w_opp_pts"] / df["w_poss"]
    # team_games.minutes is game length (48, 53 with one overtime), so this is per 48 minutes.
    pace = 48 * df["w_poss"] / df["w_min"]
    lg_rtg = 100 * df["lg_pts"] / df["lg_poss"]
    lg_pace = 48 * df["lg_poss"] / df["lg_min"]
    df["opp_def_rtg_rel"] = (def_rtg - lg_rtg).where(enough)
    df["opp_pace_rel"] = (pace - lg_pace).where(enough)
    return df[["game_id", "team", "opp_def_rtg_rel", "opp_pace_rel"]]


# ---- 3. Teammates out -----------------------------------------------------------------

def teammate_features(luka_games: pd.DataFrame, team_players: pd.DataFrame, player_id: int = PLAYER_ID) -> pd.DataFrame:
    """For each Luka game: rotation teammates from the team's previous games who didn't play.

    Rotation = played at least ROTATION_MIN_GAMES of the team's last TEAM_WINDOW games, averaging
    ROTATION_MIN_MINUTES+ when they played. Shot volume ("load") = FGA + 0.44*FTA + TOV.
    usage_share_out is the out players' average load over the team's average load per game.
    """
    tp = team_players.copy()
    tp["load"] = tp["fga"] + 0.44 * tp["fta"] + tp["tov"]
    out_rows = []
    for team, team_rows in tp.groupby("team"):
        schedule = team_rows.drop_duplicates("game_id").sort_values("game_date")[["game_id", "game_date"]]
        by_game = {g: rows for g, rows in team_rows.groupby("game_id")}
        for _, g in luka_games[luka_games["team"] == team].iterrows():
            earlier = schedule[schedule["game_date"] < g["game_date"]].tail(TEAM_WINDOW)
            if len(earlier) < TEAM_WINDOW // 2:
                out_rows.append({"game_id": g["game_id"]})
                continue
            window = pd.concat([by_game[x] for x in earlier["game_id"]])
            team_load_per_game = window["load"].sum() / len(earlier)
            mates = window[window["player_id"] != player_id].groupby("player_id").agg(
                games=("game_id", "nunique"), minutes=("minutes", "mean"), load=("load", "mean"))
            rotation = mates[(mates["games"] >= ROTATION_MIN_GAMES) & (mates["minutes"] >= ROTATION_MIN_MINUTES)]
            played = set(by_game[g["game_id"]]["player_id"]) if g["game_id"] in by_game else set()
            out = rotation[~rotation.index.isin(played)]
            top = rotation["load"].idxmax() if len(rotation) else None
            out_rows.append({
                "game_id": g["game_id"],
                "n_rotation_out": float(len(out)),
                "usage_share_out": float(out["load"].sum() / team_load_per_game) if team_load_per_game else 0.0,
                "top_teammate_out": float(top is not None and top not in played),
            })
    return pd.DataFrame(out_rows, columns=["game_id", "n_rotation_out", "usage_share_out", "top_teammate_out"])


# ---- Assemble -------------------------------------------------------------------------

def build(db, player_id: int = PLAYER_ID) -> pd.DataFrame:
    """One row per Luka game: game_id, game_date and FEATURE_COLUMNS (NaN where unknown)."""
    games = pd.DataFrame(list(db.games.find({"player_id": player_id}, {"_id": 0})))
    form = add_form_fallbacks(pd.DataFrame(list(db.games.aggregate(form_pipeline(player_id)))), games)

    opp_rows = pd.DataFrame(list(db.team_games.aggregate(opponent_pipeline(games["game_id"].tolist()))))
    opp = opponent_features(opp_rows)
    # Each Luka game has two team rows; keep the opponent's.
    opp = opp.merge(form[["game_id", "opponent"]], left_on=["game_id", "team"], right_on=["game_id", "opponent"])

    team_players = pd.DataFrame(list(db.team_player_games.find(
        {"team": {"$in": games["team"].unique().tolist()}},
        {"_id": 0, "game_id": 1, "player_id": 1, "team": 1, "game_date": 1, "minutes": 1, "fga": 1, "fta": 1, "tov": 1},
    )))
    mates = teammate_features(form[["game_id", "game_date", "team"]], team_players, player_id)

    df = (form.merge(opp[["game_id", "opp_def_rtg_rel", "opp_pace_rel"]], on="game_id", how="left")
              .merge(mates, on="game_id", how="left"))
    return df.sort_values("game_date").reset_index(drop=True)[["game_id", "game_date", "season", *FEATURE_COLUMNS]]


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 4)


def feature_docs(df: pd.DataFrame, player_id: int = PLAYER_ID, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    return [{
        "game_id": r["game_id"],
        "player_id": player_id,
        "as_of": r["game_date"].to_pydatetime(),
        "computed_at": now,
        "pipeline_version": PIPELINE_VERSION,
        "availability_source": "box_score",
        "values": {c: _clean(r[c]) for c in FEATURE_COLUMNS},
    } for _, r in df.iterrows()]


def write(db, docs: list[dict]) -> None:
    db.features.bulk_write([
        UpdateOne({"game_id": d["game_id"], "pipeline_version": d["pipeline_version"], "as_of": d["as_of"]},
                  {"$set": d}, upsert=True)
        for d in docs
    ], ordered=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="compute and summarize, write nothing")
    args = parser.parse_args()

    from luka.db.client import get_db

    db = get_db()
    df = build(db)
    print(f"{len(df)} games, pipeline {PIPELINE_VERSION}")
    print(df[FEATURE_COLUMNS].describe().T[["count", "mean", "min", "max"]].round(2).to_string())
    if not args.dry_run:
        write(db, feature_docs(df))
        print(f"wrote {len(df)} features docs")


if __name__ == "__main__":
    main()
