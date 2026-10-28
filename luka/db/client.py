"""MongoDB connection. Reads MONGODB_URI and MONGODB_DB from .env.

    python -m luka.db.client   # ping the cluster to check the connection
"""
import os
import sys
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError
from pymongo.server_api import ServerApi

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set. Copy .env.example to .env and fill it in.")
    # tz_aware: datetimes come back as UTC-aware, so as_of comparisons never mix naive and aware.
    # 10s server selection: fail fast instead of hanging 30s when the cluster is unreachable.
    return MongoClient(uri, server_api=ServerApi("1"), tz_aware=True, serverSelectionTimeoutMS=10_000)


def get_db(name: str | None = None) -> Database:
    return get_client()[name or os.environ.get("MONGODB_DB", "luka")]


def ping() -> bool:
    """Ping the cluster and print a success or failure message. Never prints the URI."""
    try:
        get_client().admin.command("ping")
    except RuntimeError as e:
        print(f"FAILED: {e}")
    except ConfigurationError as e:
        print(f"FAILED: MONGODB_URI looks malformed ({e}).")
    except OperationFailure as e:
        print(f"FAILED: cluster rejected the login. Check the user and password in MONGODB_URI. ({e.code_name})")
    except ServerSelectionTimeoutError:
        print("FAILED: could not reach the cluster within 10s. In Atlas, check Network Access "
              "allows your current IP and that the cluster is running.")
    else:
        print(f"OK: connected to Atlas; database '{get_db().name}' is ready.")
        return True
    return False


if __name__ == "__main__":
    sys.exit(0 if ping() else 1)
