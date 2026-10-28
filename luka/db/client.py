"""MongoDB connection. Reads MONGODB_URI and MONGODB_DB from .env."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.server_api import ServerApi

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set. Copy .env.example to .env and fill it in.")
    # tz_aware: datetimes come back as UTC-aware, so as_of comparisons never mix naive and aware.
    return MongoClient(uri, server_api=ServerApi("1"), tz_aware=True)


def get_db(name: str | None = None) -> Database:
    return get_client()[name or os.environ.get("MONGODB_DB", "luka")]
