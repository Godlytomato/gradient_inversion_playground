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
    Return True only for a terminal experiment of this versioned configuration.

    Error/interrupted rows are preserved and retried as new attempts. Historical
    unprefixed hashes remain a separate protocol and do not suppress v2 runs.
    """

    connection = get_connection()

    try:

        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        if not columns:
            return False
        identity = "(run_hash = ? OR config_hash = ?)" if "config_hash" in columns else "run_hash = ?"
        parameters = (run_hash, run_hash) if "config_hash" in columns else (run_hash,)
        row = connection.execute(
            f"SELECT 1 FROM runs WHERE {identity} AND outcome IN ('success', 'failure') LIMIT 1",
            parameters,
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

        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT attempt, outcome FROM runs WHERE config_hash = ? OR run_hash = ? ORDER BY attempt DESC",
            (run_data["run_hash"], run_data["run_hash"]),
        ).fetchall()
        if any(outcome in {"success", "failure"} for _, outcome in previous):
            raise ValueError("This configuration already has a completed experiment")
        attempt = max((row[0] for row in previous), default=0) + 1
        identity = run_data["run_hash"] if attempt == 1 else f'{run_data["run_hash"]}:attempt:{attempt}'
        cursor = connection.execute(
            """
            INSERT INTO runs (
                run_hash,
                config_hash,
                attempt,
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identity,
                run_data["run_hash"],
                attempt,
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
    summary: dict | None = None,
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
                best_reconstruction_hash = ?,
                summary = ?
            WHERE id = ?
            """,
            (
                outcome,
                best_reconstruction_hash,
                json.dumps(json_safe(summary or {}), allow_nan=False),
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

    metrics_json = json.dumps(json_safe(metrics), allow_nan=False)

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
        weights_only=True,
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
            weights_only=True,
        )

    return images
def json_safe(value):
    """Lossless finite JSON scalars plus explicit representations of infinity."""
    import math
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value == float('inf') else "-Infinity" if value == -float('inf') else "NaN"
    return value


def save_run_metrics(run_hash: str | None = None, metrics: dict | None = None, *, run_id: int | None = None):
    import json

    connection = get_connection()
    try:
        key = "id" if run_id is not None else "run_hash"
        cursor = connection.execute(
            f"UPDATE runs SET run_metrics = ? WHERE {key} = ?",
            (json.dumps(json_safe(metrics or {}), allow_nan=False), run_id if run_id is not None else run_hash),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Unknown run: {run_hash}")
        connection.commit()
    finally:
        connection.close()
