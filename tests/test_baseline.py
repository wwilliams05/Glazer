from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from luka.model.baseline import add_baselines, mae_table

T0 = datetime(2025, 10, 22, tzinfo=timezone.utc)


def _games(season_pts: dict[str, list[int]]) -> pd.DataFrame:
    rows, day = [], 0
    for season, pts in season_pts.items():
        for p in pts:
            rows.append({"game_id": f"G{day:03d}", "season": season, "game_date": T0 + timedelta(days=day),
                         "pts": p, "reb": 5, "ast": 8})
            day += 1
    return pd.DataFrame(rows)


def test_season_average_uses_only_earlier_games():
    df = add_baselines(_games({"2025-26": [30, 20, 40]}))
    assert pd.isna(df.loc[0, "pts_season_avg"])  # rookie-style debut: nothing to go on
    assert df.loc[1, "pts_season_avg"] == 30  # not 25: the game's own 20 is excluded
    assert df.loc[2, "pts_season_avg"] == 25


def test_first_game_of_season_falls_back_to_previous_season():
    df = add_baselines(_games({"2024-25": [30, 20], "2025-26": [50, 10]}))
    assert df.loc[2, "pts_season_avg"] == 25  # 2024-25 average
    assert df.loc[3, "pts_season_avg"] == 50  # only this season's games once there are any


def test_future_games_never_change_past_predictions():
    early = add_baselines(_games({"2025-26": [30, 20, 40]}))
    later = add_baselines(_games({"2025-26": [30, 20, 40, 99, 0]}))
    pd.testing.assert_series_equal(early["pts_season_avg"], later["pts_season_avg"].iloc[:3])


def test_mae_skips_games_without_a_prediction():
    m = mae_table(add_baselines(_games({"2025-26": [30, 20, 40]})))
    assert m["games"] == 2
    assert m["pts"] == pytest.approx((10 + 15) / 2)
    assert m["reb"] == 0
