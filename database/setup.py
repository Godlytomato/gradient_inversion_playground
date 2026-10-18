# database/setup.py

from database.connection import (
    get_connection,
    DB_PATH,
)

from database.schema import SCHEMA


def setup_database(
    reset: bool = False
):
    """
    Create the experiment database.

    If reset=True, remove the existing database first.
    """

    if reset and DB_PATH.exists():
        DB_PATH.unlink()

    connection = get_connection()

    try:
        connection.executescript(
            SCHEMA
        )

        connection.commit()

    finally:
        connection.close()


if __name__ == "__main__":
    setup_database(
        reset=True
    )

    print(
        f"Database initialized: {DB_PATH}"
    )