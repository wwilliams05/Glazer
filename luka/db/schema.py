"""Collections, validators and indexes.

Safe to re-run: missing collections are created, validators are re-applied with collMod,
and create_index is a no-op when the index already exists.

    python -m luka.db.schema            # apply
    python -m luka.db.schema --dry-run  # print the plan (works without a cluster)
"""
import argparse
import os

from pymongo import ASCENDING as ASC, DESCENDING as DESC
from pymongo.database import Database

STATS = ["pts", "reb", "ast"]
VENUES = ["kalshi", "polymarket"]

NUM = {"bsonType": "number"}  # int, long, double or decimal
NONNEG = {"bsonType": "number", "minimum": 0}
PROB = {"bsonType": "number", "minimum": 0, "maximum": 1}
STR = {"bsonType": "string"}
INT = {"bsonType": "int"}
BOOL = {"bsonType": "bool"}
DATE = {"bsonType": "date"}
OID = {"bsonType": "objectId"}
STAT = {"enum": STATS}


def _schema(required: list[str], properties: dict) -> dict:
    return {"bsonType": "object", "required": required, "properties": properties}


def _validator(required: list[str], properties: dict, *exprs: dict) -> dict:
    """$jsonSchema for shape, plus optional $expr rules that compare fields within a document."""
    rules = [{"$jsonSchema": _schema(required, properties)}]
    rules += [{"$expr": e} for e in exprs]
    return rules[0] if len(rules) == 1 else {"$and": rules}


GAMES = _validator(
    ["game_id", "player_id", "season", "game_date", "opponent", "home", "minutes", "pts", "reb", "ast"],
    {
        "game_id": STR,  # nba_api GAME_ID, e.g. "0022600045"
        "player_id": INT,
        "season": STR,  # "2026-27"
        "season_type": {"enum": ["regular", "playoffs", "play_in", "cup_final"]},
        "game_date": DATE,
        "tip_off": DATE,
        "team": STR,
        "opponent": STR,
        "home": BOOL,
        "minutes": NONNEG,
        "pts": NONNEG,
        "reb": NONNEG,
        "ast": NONNEG,
        "usage": PROB,
        "rest_days": NONNEG,
        "back_to_back": BOOL,
        "teammates_out": {
            "bsonType": "array",
            "items": _schema(["player_id", "name"], {"player_id": INT, "name": STR, "status": STR}),
        },
    },
)

FEATURES = _validator(
    ["game_id", "player_id", "as_of", "computed_at", "values"],
    {
        "game_id": STR,
        "player_id": INT,
        "as_of": DATE,  # only data timestamped before this may feed the values
        "computed_at": DATE,
        "pipeline_version": STR,
        "values": {"bsonType": "object"},
        "notes": {
            "bsonType": "array",
            "items": _schema(["text", "source_id"], {"text": STR, "source_id": OID}),
        },
    },
)

PREDICTIONS = _validator(
    ["game_id", "stat", "model_version", "as_of", "created_at", "prediction", "range", "strike_probs", "drivers"],
    {
        "game_id": STR,
        "stat": STAT,
        "model_version": STR,
        "features_id": OID,
        "as_of": DATE,
        "created_at": DATE,
        "prediction": NONNEG,
        "range": _schema(["level", "low", "high"], {"level": PROB, "low": NONNEG, "high": NONNEG}),
        "strike_probs": {
            "bsonType": "array",
            "items": _schema(["strike", "p_at_least"], {"strike": NONNEG, "p_at_least": PROB}),
        },
        "drivers": {
            "bsonType": "array",
            "maxItems": 10,
            "items": _schema(["feature", "effect"], {"feature": STR, "effect": NUM, "label": STR}),
        },
        "market": {
            "bsonType": ["object", "null"],
            "properties": {"venue": {"enum": VENUES}, "snapshot_at": DATE},
        },
        "actual": {"bsonType": ["number", "null"]},
        "error": {"bsonType": ["number", "null"]},
    },
    {"$lte": ["$range.low", "$prediction"]},
    {"$lte": ["$prediction", "$range.high"]},
    # No lookahead: the market price shown with a prediction must predate its as_of.
    {"$lte": ["$market.snapshot_at", "$as_of"]},
)

PAPER_TRADES = _validator(
    [
        "prediction_id", "game_id", "stat", "strategy_version", "venue", "market_id", "strike",
        "side", "price", "stake", "fees", "snapshot_at", "decided_at", "tip_off", "status",
    ],
    {
        "prediction_id": OID,
        "game_id": STR,
        "stat": STAT,
        "strategy_version": STR,
        "venue": {"enum": VENUES},
        "market_id": STR,
        "strike": NONNEG,
        "side": {"enum": ["yes", "no"]},
        # MongoDB's $jsonSchema is draft 4: exclusive* are boolean flags on minimum/maximum.
        "price": {"bsonType": "number", "minimum": 0, "exclusiveMinimum": True, "maximum": 1, "exclusiveMaximum": True},
        "stake": {"bsonType": "number", "minimum": 0, "exclusiveMinimum": True},
        "fees": NONNEG,
        "model_p": PROB,
        "edge": NUM,
        "snapshot_at": DATE,  # price snapshot the decision used
        "decided_at": DATE,
        "tip_off": DATE,
        "rationale": STR,
        "status": {"enum": ["open", "won", "lost", "void"]},
        "closing_price": PROB,
        "payout": NONNEG,
        "pnl": NUM,
        "settled_at": DATE,
    },
    # No lookahead: decide from an earlier price, and before the game starts.
    {"$lte": ["$snapshot_at", "$decided_at"]},
    {"$lt": ["$decided_at", "$tip_off"]},
)

STRATEGY_CONFIGS = _validator(
    ["version", "created_at", "rules"],
    {"version": STR, "created_at": DATE, "description": STR, "rules": {"bsonType": "object"}},
)

AGENT_RUNS = _validator(
    ["agent", "started_at"],
    {
        "agent": {"enum": ["chat", "research", "trading"]},
        "started_at": DATE,
        "latency_ms": NONNEG,
        "tool_calls": {"bsonType": "array"},
        "error": {"bsonType": ["string", "null"]},
    },
)

RAW_CACHE_TTL_SECONDS = 30 * 24 * 3600

# Time series collections get no unique indexes (MongoDB doesn't support them) and no
# validators; they are insert-only snapshot logs. See docs/PRD.md for the reasoning.
COLLECTIONS: dict[str, dict] = {
    "games": {
        "validator": GAMES,
        "indexes": [
            ([("game_id", ASC)], {"unique": True}),
            ([("player_id", ASC), ("game_date", DESC)], {}),
        ],
    },
    "players": {
        "indexes": [([("player_id", ASC)], {"unique": True})],
    },
    "injury_reports": {
        "timeseries": {"timeField": "reported_at", "metaField": "meta", "granularity": "minutes"},
        "indexes": [([("meta.player_id", ASC), ("reported_at", DESC)], {})],
    },
    "market_snapshots": {
        # meta: {venue, game_id, market_id, stat, strike}
        "timeseries": {"timeField": "snapshot_at", "metaField": "meta", "granularity": "minutes"},
        "indexes": [
            ([("meta.game_id", ASC), ("meta.stat", ASC), ("snapshot_at", DESC)], {}),
            ([("meta.market_id", ASC), ("snapshot_at", DESC)], {}),
        ],
    },
    "news": {
        # Vector Search index is created in week 6, through Atlas.
        "indexes": [([("published_at", DESC)], {})],
    },
    "features": {
        "validator": FEATURES,
        "indexes": [([("game_id", ASC), ("as_of", DESC)], {})],
    },
    "predictions": {
        "validator": PREDICTIONS,
        "indexes": [
            ([("game_id", ASC), ("stat", ASC), ("model_version", ASC)], {"unique": True}),
            ([("created_at", DESC)], {}),
        ],
    },
    "paper_trades": {
        "validator": PAPER_TRADES,
        "indexes": [
            # At most one bet per stat per game per strategy version.
            ([("game_id", ASC), ("stat", ASC), ("strategy_version", ASC)], {"unique": True}),
            ([("strategy_version", ASC), ("decided_at", ASC)], {}),
        ],
    },
    "strategy_configs": {
        "validator": STRATEGY_CONFIGS,
        "indexes": [([("version", ASC)], {"unique": True})],
    },
    "agent_runs": {
        "validator": AGENT_RUNS,
        "indexes": [([("agent", ASC), ("started_at", DESC)], {})],
    },
    "raw_api_cache": {
        "indexes": [
            ([("source", ASC), ("key", ASC), ("fetched_at", DESC)], {}),
            ([("fetched_at", ASC)], {"expireAfterSeconds": RAW_CACHE_TTL_SECONDS}),
        ],
    },
}

VALIDATION = {"validationLevel": "strict", "validationAction": "error"}


def ensure_schema(db: Database | None, dry_run: bool = False) -> list[str]:
    """Create or update every collection and index. Returns the actions taken (or planned).

    db may be None only for a dry run, which then plans against an empty database.
    """
    if db is None and not dry_run:
        raise ValueError("db is required unless dry_run=True")
    existing = {c["name"]: c for c in db.list_collections()} if db is not None else {}
    actions = []

    for name, spec in COLLECTIONS.items():
        ts = spec.get("timeseries")
        validator = spec.get("validator")

        if name not in existing:
            opts = {}
            if ts:
                opts["timeseries"] = ts
            if validator:
                opts.update(validator=validator, **VALIDATION)
            actions.append(f"create {name}" + (" (time series)" if ts else ""))
            if not dry_run:
                db.create_collection(name, **opts)
        else:
            is_ts = existing[name].get("type") == "timeseries"
            if is_ts != bool(ts):
                # Time series can't be converted in place; this needs a manual migration.
                want = "time series" if ts else "regular"
                raise RuntimeError(f"{name} exists but is not a {want} collection; drop or migrate it first")
            if validator:
                actions.append(f"update validator {name}")
                if not dry_run:
                    db.command("collMod", name, validator=validator, **VALIDATION)

        for keys, opts in spec["indexes"]:
            actions.append(f"index {name} {keys} {opts or ''}".rstrip())
            if not dry_run:
                db[name].create_index(keys, **opts)

    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db", help="database name (default: MONGODB_DB or 'luka')")
    args = parser.parse_args()

    from luka.db.client import get_db

    db = get_db(args.db) if not args.dry_run or os.environ.get("MONGODB_URI") else None
    for action in ensure_schema(db, dry_run=args.dry_run):
        print(action)


if __name__ == "__main__":
    main()
