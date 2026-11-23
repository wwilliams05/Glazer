"""Train and evaluate one model per stat (PTS, REB, AST) on the f1 features.

Each model predicts an adjustment to the baseline (Luka's season average to date), not the
stat itself. Ridge and gradient boosting shrink toward the average of their training
target; with this target that average is "no adjustment", so a model that learns nothing
falls back to the baseline instead of to his career average.

Time-ordered throughout (CLAUDE.md: never shuffle games across a split):

    walk-forward  2021-22 .. 2024-25   each season predicted by a model trained on the seasons
                                       before it; pooled error picks the model (baseline
                                       included as a candidate) and gives the residuals
    test          2025-26 onward       scored after every choice is made

History: the first version (m1, 2026-11-22) predicted the raw stat and picked models on
2024-25 alone; it lost to the baseline on the test games. This version (m2) was designed
after seeing that, so the test games are no longer a clean holdout. Live games from
2026-11-23 on are the clean check.

The predictive distribution comes from the walk-forward residuals (actual - prediction).
The 80% range is prediction + their 10th and 90th percentiles, and
P(stat >= k) = share of residuals > (k - 0.5 - prediction).

    python -m luka.model.train
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from luka.features.pipeline import FEATURE_COLUMNS, PIPELINE_VERSION

MODEL_VERSION = "m2"
STATS = ["pts", "reb", "ast"]
TEST_FROM = "2025-26"
WALK_FORWARD_FROM = "2021-22"  # first season predicted in the walk-forward (needs 3 seasons to train on)
RANGE_LEVEL = 0.8
SEED = 0

CANDIDATES = (
    [("baseline", {})]
    + [("ridge", {"alpha": a}) for a in (3.0, 30.0, 300.0, 3000.0)]
    + [("gbm", {"max_depth": d, "learning_rate": lr, "max_iter": n, "loss": loss})
       for d, lr, n, loss in itertools.product((2, 3), (0.03, 0.1), (100, 300), ("squared_error", "absolute_error"))]
)


def load_dataset(db) -> pd.DataFrame:
    feats = pd.DataFrame([{"game_id": d["game_id"], **d["values"]} for d in db.features.find(
        {"pipeline_version": PIPELINE_VERSION}, {"_id": 0, "game_id": 1, "values": 1})])
    games = pd.DataFrame(list(db.games.find({}, {"_id": 0, "game_id": 1, "game_date": 1, "season": 1, **{s: 1 for s in STATS}})))
    df = games.merge(feats, on="game_id").sort_values(["game_date", "game_id"]).reset_index(drop=True)
    # Rows with no history (career start, first games of the opponent windows) can't be scored fairly.
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def make_model(kind: str, params: dict):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    return HistGradientBoostingRegressor(random_state=SEED, **params)


def fit_predict(kind: str, params: dict, train: pd.DataFrame, test: pd.DataFrame, stat: str) -> np.ndarray:
    """Baseline (season average to date) plus a learned adjustment, floored at 0."""
    base = test[f"{stat}_season"].to_numpy()
    if kind == "baseline":
        return base
    adjustment = train[stat] - train[f"{stat}_season"]
    model = make_model(kind, params).fit(train[FEATURE_COLUMNS], adjustment)
    return np.clip(base + model.predict(test[FEATURE_COLUMNS]), 0, None)


def mae(y, pred) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def walk_forward(df: pd.DataFrame, stat: str, kind: str, params: dict, until: str) -> tuple[np.ndarray, np.ndarray]:
    """(actuals, predictions) for every game from WALK_FORWARD_FROM up to (not including)
    `until`, each season predicted by a model trained only on the seasons before it."""
    ys, preds = [], []
    for season in sorted(s for s in df["season"].unique() if WALK_FORWARD_FROM <= s < until):
        train, test = df[df["season"] < season], df[df["season"] == season]
        ys.append(test[stat].to_numpy())
        preds.append(fit_predict(kind, params, train, test, stat))
    return np.concatenate(ys), np.concatenate(preds)


def choose_model(df: pd.DataFrame, stat: str) -> tuple[str, dict, float, np.ndarray]:
    """Lowest pooled walk-forward MAE over every candidate, the baseline included.

    Returns (kind, params, walk-forward MAE, walk-forward residuals of the winner).
    """
    best = None
    for kind, params in CANDIDATES:
        y, pred = walk_forward(df, stat, kind, params, until=TEST_FROM)
        score = mae(y, pred)
        if best is None or score < best[2]:
            best = (kind, params, score, y - pred)
    return best


def prediction_range(pred: np.ndarray, residuals: np.ndarray, level: float = RANGE_LEVEL) -> tuple[np.ndarray, np.ndarray]:
    lo_q, hi_q = np.quantile(residuals, [(1 - level) / 2, (1 + level) / 2])
    return np.clip(pred + lo_q, 0, None), pred + hi_q


def p_at_least(pred: float, k: int, residuals: np.ndarray) -> float:
    """P(stat >= k). Stats are whole numbers, so 'at least k' means actual > k - 0.5."""
    return float(np.mean(residuals > (k - 0.5 - pred)))


def evaluate(df: pd.DataFrame, stat: str) -> dict:
    kind, params, wf_mae, residuals = choose_model(df, stat)
    pre_test, test = df[df["season"] < TEST_FROM], df[df["season"] >= TEST_FROM]
    pred = fit_predict(kind, params, pre_test, test, stat)
    lo, hi = prediction_range(pred, residuals)
    y = test[stat].to_numpy()
    baseline_wf = mae(*walk_forward(df, stat, "baseline", {}, until=TEST_FROM))
    return {
        "stat": stat, "model": kind, "params": params, "wf_mae": wf_mae, "wf_baseline_mae": baseline_wf,
        "test_games": len(test),
        "test_mae": mae(y, pred),
        "baseline_mae": mae(y, test[f"{stat}_season"]),
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "range_width": float(np.mean(hi - lo)),
        "n_residuals": len(residuals),
    }


def main() -> None:
    from luka.db.client import get_db

    df = load_dataset(get_db())
    print(f"{len(df)} games with full features; test = {TEST_FROM} onward\n")
    rows = [evaluate(df, s) for s in STATS]
    print(f"{'':<14}{'walk-forward MAE':^20}{'test MAE':^20}")
    print(f"{'stat':<5}{'model':<9}{'model':>9}{'baseline':>10}{'model':>9}{'baseline':>10}{'gain':>8}"
          f"{'80% cover':>11}{'width':>8}")
    for r in rows:
        gain = r["baseline_mae"] - r["test_mae"]
        print(f"{r['stat']:<5}{r['model']:<9}{r['wf_mae']:>9.2f}{r['wf_baseline_mae']:>10.2f}{r['test_mae']:>9.2f}"
              f"{r['baseline_mae']:>10.2f}{gain:>+8.2f}{r['coverage']:>10.0%}{r['range_width']:>8.1f}")
    for r in rows:
        print(f"  {r['stat']}: {r['params']}  ({r['test_games']} test games, {r['n_residuals']} residuals)")


if __name__ == "__main__":
    main()
