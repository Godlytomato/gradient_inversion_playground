import sqlite3
from pathlib import Path


DATABASE_DIR = Path(__file__).resolve().parent

DB_PATH = (
    DATABASE_DIR
    / "playground.db"
)


def get_connection():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.execute(
        "PRAGMA foreign_keys = ON;"
    )

    return connection