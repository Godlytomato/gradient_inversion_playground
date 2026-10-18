
from pathlib import Path
import io
import sqlite3

import torch
import matplotlib.pyplot as plt

from database.connection import DB_PATH


OUTPUT_DIR = Path("temp_images")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_tensor_image(
    image: torch.Tensor,
    output_path: Path
):
    image = image.detach().cpu()

    if (
        image.ndim == 3
        and image.shape[0] in (1, 3)
    ):
        image = image.permute(
            1,
            2,
            0
        )

    image = image.squeeze()

    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:
        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

    plt.figure()

    plt.imshow(
        image,
        cmap="gray"
        if image.ndim == 2
        else None,
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()


def export_images():

    connection = sqlite3.connect(
        DB_PATH
    )

    rows = connection.execute(
        """
        SELECT
            runs.run_hash,
            images.image_hash,
            images.image_type,
            images.image_data
        FROM images

        JOIN runs
            ON images.run_id = runs.id

        ORDER BY
            runs.run_hash,
            images.image_type
        """
    ).fetchall()

    connection.close()

    print(
        f"Found {len(rows)} stored images"
    )

    for (
        run_hash,
        image_hash,
        image_type,
        image_data,
    ) in rows:

        buffer = io.BytesIO(
            image_data
        )

        tensor = torch.load(
            buffer,
            map_location="cpu",
            weights_only=False,
        )

        if tensor.ndim == 4:

            for index in range(
                tensor.shape[0]
            ):

                output_path = OUTPUT_DIR / (
                    f"{run_hash[:12]}"
                    f"__{image_type}"
                    f"__{index}.png"
                )

                save_tensor_image(
                    tensor[index],
                    output_path,
                )

                print(
                    f"Saved: {output_path}"
                )

        else:

            output_path = OUTPUT_DIR / (
                f"{run_hash[:12]}"
                f"__{image_type}.png"
            )

            save_tensor_image(
                tensor,
                output_path,
            )

            print(
                f"Saved: {output_path}"
            )


if __name__ == "__main__":
    export_images()