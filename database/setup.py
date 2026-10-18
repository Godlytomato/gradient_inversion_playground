# database/setup.py

from database.connection import (
    get_connection,
    DB_PATH,
)

from database.schema import migrate_schema


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
        migrate_schema(connection)

    finally:
        connection.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Initialize or migrate the experiment database")
    parser.add_argument("--reset", action="store_true", help="Delete the existing database and all experiment results")
    args = parser.parse_args(argv)
    setup_database(reset=args.reset)

    print(
        f"Database initialized: {DB_PATH}"
    )


if __name__ == "__main__":
    main()
