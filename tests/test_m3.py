import numpy as np
import pandas as pd
import pytest

from luka.model import m3
from luka.model.m3 import (
    COMPONENTS, X_COLUMNS, candidate_predict, drivers, fit_candidate, load_card, rps, walk_forward,
)


def _df(seasons=("2019-20", "2020-21", "2021-22", "2022-23"), n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for i in range(n):
            r = {c: rng.normal() for c in X_COLUMNS}
            r.update(season=s, game_id=f"{s}-{i}", pts_season=30.0, pts_ew10=32.0, reb_season=8.0, reb_ew10=8.0,
                     ast_season=8.0, ast_ew10=8.0)
            r["pts"] = 31 + rng.normal()
            rows.append(r)
    return pd.DataFrame(rows)


def test_ensemble_is_the_mean_of_its_members():
    df = _df()
    fitted = fit_candidate(("season", "ew"), df, "pts")
    assert candidate_predict(fitted, df.iloc[:1], "pts")[0] == pytest.approx((30 + 32) / 2)


def test_walk_forward_never_trains_on_the_season_it_predicts(monkeypatch):
    seen = []
    real = m3.fit_component

    def spy(name, train, stat):
        seen.append(set(train["season"]))
        return real(name, train, stat)

    monkeypatch.setattr(m3, "fit_component", spy)
    df = _df()
    y, _ = walk_forward(df, "pts", until="2023-24")
    assert len(y) == 2 * 30  # 2021-22 and 2022-23 are predicted
    assert all(seasons <= {"2019-20", "2020-21", "2021-22"} for seasons in seen)
    assert {"2019-20", "2020-21"} in seen and {"2019-20", "2020-21", "2021-22"} in seen


def test_drivers_split_the_gap_to_the_season_average():
    # season + ew ensemble: prediction - season average = (ew - season) / 2, all credited to recent form.
    df = _df()
    fitted = fit_candidate(("season", "ew"), df, "pts")
    x = df.iloc[0]
    d = drivers(fitted, x, df, "pts", labels={})
    assert d == [{"feature": "pts_ew10", "label": "Points, recent form (weighted)", "value": 32.0, "effect": 1.0}]


def test_rps_rewards_sharp_correct_and_punishes_confident_wrong():
    y = np.array([30])
    narrow = rps(np.array([30.0]), y, np.array([-1.0, 0.0, 1.0]))
    wide = rps(np.array([30.0]), y, np.linspace(-15, 15, 31))
    wrong = rps(np.array([20.0]), y, np.array([-1.0, 0.0, 1.0]))
    assert narrow < wide < wrong


def test_card_from_other_components_is_refused(tmp_path):
    card = load_card()
    card["components"][card["stats"]["pts"]["members"][0]]["params"] = {"alpha": 1.0}
    p = tmp_path / "m3.json"
    p.write_text(__import__("json").dumps(card), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_card(p)


def test_no_component_reads_teammate_features():
    for name, (_, _, _, inputs) in COMPONENTS.items():
        assert not set(m3.INPUTS[inputs]) & set(m3.MATES), name
