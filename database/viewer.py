

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from database.connection import get_connection
from database.database import get_run_images


if len(sys.argv) != 2:
    raise ValueError(
        "Usage: python -m database.viewer <run_hash>"
    )


RUN_HASH = sys.argv[1]


OUTPUT_DIR = Path(
    "temp_view"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def get_run_id(
    run_hash: str
) -> int:

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT id
            FROM runs
            WHERE run_hash = ?
            """,
            (
                run_hash,
            )
        ).fetchone()

    finally:

        connection.close()

    if row is None:
        raise KeyError(
            f"Unknown run hash: {run_hash}"
        )

    return row[0]


run_id = get_run_id(
    RUN_HASH
)

images = get_run_images(
    run_id
)


if "ground_truth" not in images:
    raise KeyError(
        "Run does not contain ground_truth image"
    )


if "best_reconstruction" not in images:
    raise KeyError(
        "Run does not contain best_reconstruction image"
    )


original = images[
    "ground_truth"
]

reconstruction = images[
    "best_reconstruction"
]


MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465]
).view(
    3,
    1,
    1
)

STD = torch.tensor(
    [0.2470, 0.2435, 0.2616]
).view(
    3,
    1,
    1
)


def viewable(
    image
):

    image = (
        image
        .detach()
        .cpu()
    )

    # undo CIFAR normalization
    if (
        image.ndim == 3
        and image.shape[0] == 3
    ):

        image = (
            image * STD
            + MEAN
        )

    image = image.clamp(
        0.0,
        1.0
    )

    if image.ndim == 3:

        image = image.permute(
            1,
            2,
            0
        )

    return image


for index in range(
    original.shape[0]
):

    fig, axes = plt.subplots(
        1,
        2
    )

    orig = viewable(
        original[index]
    )

    recon = viewable(
        reconstruction[index]
    )

    axes[0].imshow(
        orig.squeeze()
    )

    axes[0].set_title(
        "Original"
    )

    axes[0].axis(
        "off"
    )


    axes[1].imshow(
        recon.squeeze()
    )

    axes[1].set_title(
        "Best Reconstruction"
    )

    axes[1].axis(
        "off"
    )


    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{RUN_HASH[:12]}_{index}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"Saved: {output_file}"
    )