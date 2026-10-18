import hashlib
import io
import json

import torch

from database.connection import get_connection
from database.serialization import (
    serialize_run,
    serialize_attack_components,
    serialize_defense_components,
)


# ============================================================
# run helpers
# ============================================================

def run_exists(
    run_hash: str,
) -> bool:
    """
    Return True if a run with this hash already exists.
    """

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT 1
            FROM runs
            WHERE run_hash = ?
            LIMIT 1
            """,
            (
                run_hash,
            ),
        ).fetchone()

        return row is not None

    finally:
        connection.close()


def create_run(
    description,
) -> int:
    """
    Create a run and store its attack and defense components.
    """

    run_data = serialize_run(
        description
    )

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            INSERT INTO runs (
                run_hash,
                name,
                model,
                dataset,
                batch_size,
                attack_hash,
                attack_name,
                defense_hash,
                defense_name,
                raw_description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_data["run_hash"],
                run_data["name"],
                run_data["model"],
                run_data["dataset"],
                run_data["batch_size"],
                run_data["attack_hash"],
                run_data["attack_name"],
                run_data["defense_hash"],
                run_data["defense_name"],
                run_data["raw_description"],
            ),
        )

        run_id = cursor.lastrowid

        attack_components = (
            serialize_attack_components(
                run_id=run_id,
                attack=description.attack,
            )
        )

        defense_components = (
            serialize_defense_components(
                run_id=run_id,
                transformations=(
                    description.gradient_transformations
                ),
            )
        )

        for component in (
            attack_components
            + defense_components
        ):

            add_component(
                connection=connection,
                **component,
            )

        connection.commit()

        return run_id

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def finish_run(
    run_id: int,
    outcome: str,
    best_reconstruction_hash: str | None = None,
):
    """
    Store the final outcome and best reconstruction for a run.
    """

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            UPDATE runs
            SET
                outcome = ?,
                best_reconstruction_hash = ?
            WHERE id = ?
            """,
            (
                outcome,
                best_reconstruction_hash,
                run_id,
            ),
        )

        if cursor.rowcount != 1:
            raise RuntimeError(
                f"Could not finish run: {run_id}"
            )

        connection.commit()

    finally:
        connection.close()


# ============================================================
# components
# ============================================================

def add_component(
    connection,
    run_id: int,
    category: str,
    name: str,
    position: int = 0,
    parameters: str | None = None,
):
    """
    Add one serialized attack or defense component.
    """

    connection.execute(
        """
        INSERT INTO components (
            run_id,
            category,
            name,
            position,
            parameters
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            run_id,
            category,
            name,
            position,
            parameters,
        ),
    )


# ============================================================
# logs
# ============================================================

def add_log(
    run_id: int,
    step: int,
    metrics: dict,
):
    """
    Store one iteration's metrics.
    """

    metrics_json = json.dumps(
        metrics,
        default=str,
    )

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO logs (
                run_id,
                step,
                metrics
            )
            VALUES (?, ?, ?)
            """,
            (
                run_id,
                step,
                metrics_json,
            ),
        )

        connection.commit()

    finally:
        connection.close()


# ============================================================
# images
# ============================================================

def add_image(
    run_id: int,
    image: torch.Tensor,
    image_type: str = "reconstruction",
) -> str:
    """
    Serialize and store an image tensor.
    """

    image = (
        image
        .detach()
        .cpu()
    )

    buffer = io.BytesIO()

    torch.save(
        image,
        buffer,
    )

    image_bytes = (
        buffer.getvalue()
    )

    image_hash = hashlib.sha256(
        str(
            run_id
        ).encode("utf-8")
        + image_type.encode("utf-8")
        + image_bytes
    ).hexdigest()

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO images (
                image_hash,
                run_id,
                image_type,
                image_data
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                image_hash,
                run_id,
                image_type,
                image_bytes,
            ),
        )

        connection.commit()

    finally:
        connection.close()

    return image_hash


def get_image(
    image_hash: str,
) -> torch.Tensor:
    """
    Load one stored image tensor by image hash.
    """

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT image_data
            FROM images
            WHERE image_hash = ?
            """,
            (
                image_hash,
            ),
        ).fetchone()

    finally:
        connection.close()

    if row is None:
        raise KeyError(
            f"Unknown image hash: {image_hash}"
        )

    buffer = io.BytesIO(
        row[0]
    )

    return torch.load(
        buffer,
        map_location="cpu",
        weights_only=False,
    )


def get_run_images(
    run_id: int,
) -> dict[str, torch.Tensor]:
    """
    Load all stored images for a run.
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT
                image_type,
                image_data
            FROM images
            WHERE run_id = ?
            """,
            (
                run_id,
            ),
        ).fetchall()

    finally:
        connection.close()

    images = {}

    for (
        image_type,
        image_bytes,
    ) in rows:

        buffer = io.BytesIO(
            image_bytes
        )

        images[
            image_type
        ] = torch.load(
            buffer,
            map_location="cpu",
            weights_only=False,
        )

    return images
def save_run_metrics(run_hash: str, metrics: dict):
    import json

    connection = get_connection()
    try:
        cursor = connection.execute(
            "UPDATE runs SET run_metrics = ? WHERE run_hash = ?",
            (json.dumps(metrics), run_hash),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Unknown run: {run_hash}")
        connection.commit()
    finally:
        connection.close()