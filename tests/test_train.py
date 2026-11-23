import numpy as np
import pandas as pd
import pytest

from luka.features.pipeline import FEATURE_COLUMNS
from luka.model.train import fit_predict, p_at_least, prediction_range, walk_forward

SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]


def _df(seed: int = 0, games_per_season: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, season in enumerate(SEASONS):
        for g in range(games_per_season):
            feats = {c: rng.normal() for c in FEATURE_COLUMNS}
            feats.update(pts_season=28.0, reb_season=8.0, ast_season=8.0)
            rows.append({"season": season, "game_date": pd.Timestamp("2018-10-01") + pd.Timedelta(days=400 * i + g),
                         "pts": 28 + rng.normal(0, 6), "reb": 8.0, "ast": 8.0, **feats})
    return pd.DataFrame(rows)


def test_baseline_candidate_returns_season_average():
    df = _df()
    pred = fit_predict("baseline", {}, df.iloc[:10], df.iloc[10:20], "pts")
    assert np.allclose(pred, 28.0)


def test_model_that_learns_nothing_falls_back_to_baseline():
    df = _df()  # reb always equals its season average: the adjustment to learn is exactly 0
    pred = fit_predict("ridge", {"alpha": 1.0}, df.iloc[:100], df.iloc[100:120], "reb")
    assert np.allclose(pred, 8.0)


def test_walk_forward_never_trains_on_the_season_it_predicts():
    df = _df()
    y1, p1 = walk_forward(df, "pts", "ridge", {"alpha": 1.0}, until="2023-24")
    later = df.copy()
    last = later["season"] == SEASONS[-1]
    later.loc[last, "pts"] = 500.0  # wildly different outcomes in the final season
    y2, p2 = walk_forward(later, "pts", "ridge", {"alpha": 1.0}, until="2023-24")
    n_before_last = int((~last & (df["season"] >= "2021-22")).sum())
    # Predictions for earlier seasons are untouched; the final season's predictions too,
    # because its own outcomes are never in its training data.
    assert np.allclose(p1[:n_before_last], p2[:n_before_last])
    assert np.allclose(p1[n_before_last:], p2[n_before_last:])


def test_prediction_range_uses_residual_percentiles():
    residuals = np.arange(-50, 51, dtype=float)  # 10th pct = -40, 90th = +40
    lo, hi = prediction_range(np.array([30.0]), residuals)
    assert lo[0] == pytest.approx(0.0)  # 30 - 40 floors at 0
    assert hi[0] == pytest.approx(70.0)


def test_p_at_least_counts_residuals_above_the_half_point():
    residuals = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert p_at_least(30.0, 30, residuals) == pytest.approx(0.6)  # actual > 29.5: residual > -0.5
    assert p_at_least(30.0, 33, residuals) == pytest.approx(0.2)  # residual > 2.5: only 3.0, 1 of 5
