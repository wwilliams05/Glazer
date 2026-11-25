"""m3: per-stat ensembles of a season-average model and a recency-weighted model, no teammate features.

Why this design (exploration on games before 2026-11-23; notes in docs/PLAN.md, "m3"):

- Luka's season-to-date average is a noisy baseline, worst early in a season. An
  exponentially weighted average over his earlier games (10-game half-life, across seasons,
  f2's *_ew10 columns) beat it in the walk-forward for all three stats.
- The teammates-out features added nothing in the walk-forward, even though historical rows
  knew who actually played. Live rows can't know that (the calibration leak the lookahead
  review found in m2: residuals measured with perfect availability are too narrow for live
  games). m3 leaves them out, so its residuals describe the errors it will make live.
- Single models disagreed between the walk-forward and the 2025-26 test games. An equal-weight
  average of one model from each family was best or close to best on both, the usual reason
  to ensemble: their errors partly cancel.

Components, each "baseline + learned adjustment" like m2 (see COMPONENTS). Candidates are
every component alone plus a few ensembles (equal-weight means). The lowest pooled
walk-forward MAE (2021-22 to 2024-25, each season predicted by a fit on the seasons before it)
wins per stat. The range and P(stat >= k) come from the winner's walk-forward residuals.

The card is built from games before DEV_CUTOFF only. m3's clean holdout is every game it
predicts live before tip-off, from its first live run on (2026-11-27). Never backfill m3 for a
game already played and count it. The 2025-26 test numbers are optimistic: the candidate list
was shaped while they were visible. Live, the chosen candidate is refit on every finished game
before the one it predicts (same as m2).

    python -m luka.model.m3                # candidate table, test metrics vs m2
    python -m luka.model.m3 --save-card    # write luka/model/cards/m3.json (once)
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from luka.config import PLAYER_ID
from luka.features.pipeline import EW_COLUMNS, EW_HALF_LIFE, FEATURE_COLUMNS, PIPELINE_VERSION_F2, build
from luka.model.train import make_model, mae, p_at_least, prediction_range

MODEL_VERSION = "m3"
PIPELINE_VERSION = PIPELINE_VERSION_F2
STATS = ["pts", "reb", "ast"]
DEV_CUTOFF = datetime(2026, 11, 23, tzinfo=timezone.utc)  # first live (clean holdout) game day
WALK_FORWARD_FROM = "2021-22"
TEST_FROM = "2025-26"
RANGE_LEVEL = 0.8
CARD_PATH = Path(__file__).parent / "cards" / f"{MODEL_VERSION}.json"

MATES = ["n_rotation_out", "usage_share_out", "top_teammate_out"]
F1_X = [c for c in FEATURE_COLUMNS if c not in MATES]  # m2's inputs without the teammate features
X_COLUMNS = F1_X + EW_COLUMNS  # everything m3 reads; EW-family models use all of it
INPUTS = {"f1": F1_X, "f2": X_COLUMNS}

GBM = {"max_depth": 2, "learning_rate": 0.03, "max_iter": 100, "loss": "squared_error"}
# name -> (baseline, model kind or None, params, inputs). The adjustment model learns
# stat - baseline. Season-family models see m2's inputs (minus teammates); EW-family models
# also see the EW columns. Boosting got worse with the EW columns added (more inputs than
# ~400 training games support), so each family keeps its own inputs.
COMPONENTS = {
    "season": ("season", None, {}, "f1"),
    "season_gbm": ("season", "gbm", GBM, "f1"),
    "season_ridge": ("season", "ridge", {"alpha": 300.0}, "f1"),
    "ew": ("ew", None, {}, "f2"),
    "ew_ridge300": ("ew", "ridge", {"alpha": 300.0}, "f2"),
    "ew_ridge3000": ("ew", "ridge", {"alpha": 3000.0}, "f2"),
}
CANDIDATES = [(c,) for c in COMPONENTS] + [
    ("season", "ew"),
    ("season_gbm", "ew_ridge300"),
    ("season_ridge", "ew_ridge300"),
    ("season_gbm", "ew_ridge300", "ew"),
]


def base_column(base: str, stat: str) -> str:
    return f"{stat}_season" if base == "season" else f"{stat}_ew{EW_HALF_LIFE}"


# ---- Data -----------------------------------------------------------------------------

def load_frame(db, until: datetime | None = DEV_CUTOFF) -> pd.DataFrame:
    """f2 features plus actuals for every finished Luka game before `until` (None: all)."""
    feats = build(db, PLAYER_ID)
    games = pd.DataFrame(list(db.games.find({"player_id": PLAYER_ID}, {"_id": 0, "game_id": 1, **{s: 1 for s in STATS}})))
    df = feats.merge(games, on="game_id")  # inner: finished games only
    if until is not None:
        df = df[df["game_date"] < until]
    return df.dropna(subset=X_COLUMNS).sort_values(["game_date", "game_id"]).reset_index(drop=True)


# ---- Components and candidates --------------------------------------------------------

def fit_component(name: str, train: pd.DataFrame, stat: str):
    base, kind, params, inputs = COMPONENTS[name]
    if kind is None:
        return None
    b = base_column(base, stat)
    return make_model(kind, params).fit(train[INPUTS[inputs]], train[stat] - train[b])


def component_predict(name: str, model, rows: pd.DataFrame, stat: str) -> np.ndarray:
    base, _, _, inputs = COMPONENTS[name]
    b = rows[base_column(base, stat)].to_numpy(dtype=float)
    adj = 0.0 if model is None else model.predict(rows[INPUTS[inputs]].astype(float))
    return np.clip(b + adj, 0, None)


def fit_candidate(members: tuple, train: pd.DataFrame, stat: str) -> list:
    return [(m, fit_component(m, train, stat)) for m in members]


def candidate_predict(fitted: list, rows: pd.DataFrame, stat: str) -> np.ndarray:
    return np.mean([component_predict(m, model, rows, stat) for m, model in fitted], axis=0)


def walk_forward(df: pd.DataFrame, stat: str, until: str = TEST_FROM) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(actuals, {component: predictions}) for seasons WALK_FORWARD_FROM..until (exclusive),
    each season predicted by components fit only on the seasons before it."""
    ys, preds = [], {c: [] for c in COMPONENTS}
    for season in sorted(s for s in df["season"].unique() if WALK_FORWARD_FROM <= s < until):
        train, test = df[df["season"] < season], df[df["season"] == season]
        ys.append(test[stat].to_numpy())
        for c in COMPONENTS:
            preds[c].append(component_predict(c, fit_component(c, train, stat), test, stat))
    return np.concatenate(ys), {c: np.concatenate(p) for c, p in preds.items()}


def choose(df: pd.DataFrame, stat: str) -> tuple[tuple, float, np.ndarray, list[tuple]]:
    """Best candidate by pooled walk-forward MAE. Returns (members, MAE, residuals, table)."""
    y, comp = walk_forward(df[df["season"] < TEST_FROM], stat)
    table = []
    for members in CANDIDATES:
        pred = np.mean([comp[m] for m in members], axis=0)
        table.append((members, mae(y, pred), y - pred))
    best = min(table, key=lambda t: t[1])
    return best[0], best[1], best[2], [(m, s) for m, s, _ in table]


# ---- Scoring --------------------------------------------------------------------------

def rps(pred: np.ndarray, y: np.ndarray, residuals: np.ndarray, k_max: int = 90) -> float:
    """Ranked probability score: mean over games of sum_k (P(stat >= k) - [actual >= k])^2.

    Rewards probabilities that are both calibrated and sharp; lower is better. It is the
    discrete version of the CRPS and covers every strike a market could list.
    """
    k = np.arange(1, k_max + 1)
    total = 0.0
    for p, actual in zip(pred, y):
        probs = np.array([p_at_least(p, kk, residuals) for kk in k])
        total += np.sum((probs - (actual >= k)) ** 2)
    return total / len(y)


# ---- Card -----------------------------------------------------------------------------

def model_card(df: pd.DataFrame) -> dict:
    stats = {}
    for stat in STATS:
        members, wf_mae, residuals, _ = choose(df, stat)
        stats[stat] = {"members": list(members), "wf_mae": round(wf_mae, 4),
                       "residuals": [round(float(r), 4) for r in residuals]}
    return {
        "model_version": MODEL_VERSION, "pipeline_version": PIPELINE_VERSION, "range_level": RANGE_LEVEL,
        "dev_cutoff": DEV_CUTOFF.isoformat(), "x_columns": X_COLUMNS,
        "components": {k: {"baseline": b, "kind": kind, "params": p, "inputs": i} for k, (b, kind, p, i) in COMPONENTS.items()},
        "chosen_on": f"walk-forward {WALK_FORWARD_FROM} to the season before {TEST_FROM}, games before {DEV_CUTOFF:%Y-%m-%d}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": stats,
    }


def load_card(path: Path = CARD_PATH) -> dict:
    card = json.loads(path.read_text(encoding="utf-8"))
    if card["model_version"] != MODEL_VERSION or card["pipeline_version"] != PIPELINE_VERSION:
        raise RuntimeError(f"{path.name} is for {card['model_version']}/{card['pipeline_version']}")
    code = {k: {"baseline": b, "kind": kind, "params": p, "inputs": i} for k, (b, kind, p, i) in COMPONENTS.items()}
    if card["x_columns"] != X_COLUMNS or any(card["components"][m] != code.get(m)
                                             for spec in card["stats"].values() for m in spec["members"]):
        raise RuntimeError(f"{path.name} doesn't match the code's features or components; that is a new model version")
    return card


# ---- Live prediction for one game -----------------------------------------------------

FEATURE_LABEL_EXTRA = {f"{s}_ew{EW_HALF_LIFE}": f"{n}, recent form (weighted)"
                       for s, n in (("pts", "Points"), ("reb", "Rebounds"), ("ast", "Assists"), ("minutes", "Minutes"))}


def drivers(fitted: list, x: pd.Series, train: pd.DataFrame, stat: str, labels: dict, n: int = 5,
            min_effect: float = 0.05) -> list[dict]:
    """Why the prediction differs from the season average, split by input.

    prediction - season average = mean over members of (baseline - season average)
                                  + mean over members of the learned adjustment.
    The first part is credited to the recent-form column (for EW members). The second is split
    like m2: adjustment(x) - adjustment(x with that one input at its training average).
    """
    w = 1 / len(fitted)
    effects: dict[str, float] = {}
    season = float(x[f"{stat}_season"])
    for name, model in fitted:
        base, _, _, inputs = COMPONENTS[name]
        col = base_column(base, stat)
        if col != f"{stat}_season":
            effects[col] = effects.get(col, 0.0) + w * (float(x[col]) - season)
        if model is None:
            continue
        X = pd.DataFrame([x[INPUTS[inputs]].astype(float)])
        full = float(model.predict(X)[0])
        for c in INPUTS[inputs]:
            X2 = X.copy()
            X2[c] = train[c].mean()
            effects[c] = effects.get(c, 0.0) + w * (full - float(model.predict(X2)[0]))
    ranked = sorted(effects.items(), key=lambda e: -abs(e[1]))
    return [{"feature": c, "label": labels.get(c, FEATURE_LABEL_EXTRA.get(c, c)), "value": round(float(x[c]), 3),
             "effect": round(e, 2)} for c, e in ranked[:n] if abs(e) >= min_effect]


def predict_stat(stat: str, spec: dict, train: pd.DataFrame, x: pd.Series, labels: dict) -> dict:
    """Same output shape as predict.predict_stat, so prediction_docs works for both versions."""
    residuals = np.asarray(spec["residuals"])
    fitted = fit_candidate(tuple(spec["members"]), train, stat)
    pred = float(candidate_predict(fitted, pd.DataFrame([x]), stat)[0])
    lo, hi = prediction_range(np.array([pred]), residuals)
    return {
        "stat": stat, "baseline": round(float(x[f"{stat}_season"]), 2), "prediction": round(pred, 2),
        "low": round(float(lo[0]), 2), "high": round(float(hi[0]), 2),
        "model": {"kind": "ensemble" if len(fitted) > 1 else spec["members"][0], "members": spec["members"]},
        "drivers": drivers(fitted, x, train, stat, labels), "residuals": residuals,
    }


# ---- Report ---------------------------------------------------------------------------

def report(df: pd.DataFrame) -> None:
    """Candidate table, then the winners on the test games (2025-26 to DEV_CUTOFF).

    The test games are not a clean holdout (m2 was designed after seeing them), so the test
    columns are a sanity check, not proof. The clean check is live games. m2's numbers on the
    same games are in docs/PLAN.md (week 2 results).
    """
    pre, test = df[df["season"] < TEST_FROM], df[df["season"] >= TEST_FROM]
    print(f"{len(df)} games before {DEV_CUTOFF:%Y-%m-%d}; walk-forward {WALK_FORWARD_FROM}..{TEST_FROM} (excl.), "
          f"test {len(test)} games\n")
    for stat in STATS:
        members, wf_mae, residuals, table = choose(df, stat)
        print(f"{stat}: walk-forward MAE by candidate")
        for m, s in sorted(table, key=lambda t: t[1]):
            print(f"   {s:6.3f}  {' + '.join(m)}{'   <- chosen' if m == members else ''}")
        pred = candidate_predict(fit_candidate(members, pre, stat), test, stat)
        y = test[stat].to_numpy()
        lo, hi = prediction_range(pred, residuals)
        base = test[f"{stat}_season"].to_numpy()
        print(f"   test: MAE {mae(y, pred):.3f} (season average {mae(y, base):.3f}), 80% coverage "
              f"{np.mean((y >= lo) & (y <= hi)):.0%}, width {np.mean(hi - lo):.1f}, RPS {rps(pred, y, residuals):.3f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-card", action="store_true", help=f"write {CARD_PATH.name} for live predictions")
    args = parser.parse_args()

    from luka.db.client import get_db

    df = load_frame(get_db())
    if args.save_card:
        if CARD_PATH.exists():
            raise SystemExit(f"{CARD_PATH} exists; a changed model is a new MODEL_VERSION, not a new card")
        card = model_card(df)
        CARD_PATH.write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
        for s, c in card["stats"].items():
            print(f"{s}: {' + '.join(c['members'])}  walk-forward MAE {c['wf_mae']}  ({len(c['residuals'])} residuals)")
        print(f"wrote {CARD_PATH}")
        return
    report(df)


if __name__ == "__main__":
    main()
