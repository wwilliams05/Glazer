"""Season-average baseline: the error any real model has to beat.

For every game, the "prediction" is Luka's average in that season over the games he had
already played. The current game is never included (shift(1) below), so there is no
lookahead. A season's first game has no earlier games, so it falls back to the previous
season's full average, which is also known before tip-off. His rookie debut has neither
and is skipped.

MAE (mean absolute error) is the average size of the miss: an MAE of 6.0 for PTS means the
baseline is off by 6 points in a typical game.

    python -m luka.model.baseline
"""
import pandas as pd

STATS = ["pts", "reb", "ast"]
RECENT_SEASONS = 2  # also report the last N seasons, the likely held-out window in week 2


def load_games(db) -> pd.DataFrame:
    cursor = db.games.find({}, {"_id": 0, "game_id": 1, "season": 1, "game_date": 1, **{s: 1 for s in STATS}})
    return pd.DataFrame(list(cursor)).sort_values(["game_date", "game_id"]).reset_index(drop=True)


def add_baselines(games: pd.DataFrame) -> pd.DataFrame:
    """Adds <stat>_season_avg and <stat>_last10 columns, each using only earlier games."""
    df = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(df["season"].unique())
    prev_season = dict(zip(seasons[1:], seasons[:-1]))
    for stat in STATS:
        # shift(1) moves each value down a row, so game n sees games 1..n-1 only.
        to_date = df.groupby("season")[stat].transform(lambda s: s.shift(1).expanding().mean())
        full_season = df.groupby("season")[stat].mean()
        fallback = df["season"].map(prev_season).map(full_season)
        df[f"{stat}_season_avg"] = to_date.fillna(fallback)
        # A second reference point: the last 10 games, across season boundaries.
        df[f"{stat}_last10"] = df[stat].shift(1).rolling(10, min_periods=1).mean()
    return df


def mae_table(df: pd.DataFrame, method: str = "season_avg") -> dict[str, float]:
    out = {}
    for stat in STATS:
        pred = df[f"{stat}_{method}"]
        mask = pred.notna()
        out[stat] = float((df.loc[mask, stat] - pred[mask]).abs().mean())
    out["games"] = int(df[f"pts_{method}"].notna().sum())
    return out


def report(df: pd.DataFrame) -> str:
    recent = sorted(df["season"].unique())[-RECENT_SEASONS:]
    rows = [
        ("Season average, all games", mae_table(df)),
        (f"Season average, {recent[0]} on", mae_table(df[df["season"].isin(recent)])),
        ("Last 10 games, all games", mae_table(df, "last10")),
        (f"Last 10 games, {recent[0]} on", mae_table(df[df["season"].isin(recent)], "last10")),
    ]
    lines = [f"{'MAE':<30}{'PTS':>7}{'REB':>7}{'AST':>7}{'games':>8}"]
    for name, m in rows:
        lines.append(f"{name:<30}{m['pts']:>7.2f}{m['reb']:>7.2f}{m['ast']:>7.2f}{m['games']:>8}")
    return "\n".join(lines)


def main() -> None:
    from luka.db.client import get_db

    games = load_games(get_db())
    print(report(add_baselines(games)))


if __name__ == "__main__":
    main()
