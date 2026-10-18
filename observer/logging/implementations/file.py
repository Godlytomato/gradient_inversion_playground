import csv
import json
import os
from dataclasses import asdict, is_dataclass

import torch
from torchvision.utils import save_image
from database.database import json_safe
from database.image_display import display_tensor, align_for_display


class file_logger:
    def __init__(
        self,
        metrics,
        run_id,
        path="results",
        description=None,
    ):
        self.metrics = metrics
        self.run_id = run_id

        self.directory = os.path.join(
            path,
            run_id
        )

        # allow rerunning same configuration
        os.makedirs(
            self.directory,
            exist_ok=True
        )

        self.csv_path = os.path.join(
            self.directory,
            "metrics.csv"
        )

        self.header_path = os.path.join(
            self.directory,
            "header.json"
        )

        self.file = open(
            self.csv_path,
            "w",
            newline=""
        )

        self.writer = csv.writer(
            self.file
        )

        self.writer.writerow(
            ["step", *metrics]
        )


    def log(
        self,
        step,
        metrics
    ):
        row = [step]

        for name in self.metrics:
            row.append(
                metrics.get(name)
            )

        self.writer.writerow(
            row
        )

        self.file.flush()


    def save_summary(
        self,
        description,
        summary,
        best_image=None,
        ground_truth=None,
        oracle_image=None,
    ):

        if is_dataclass(description):
            description = asdict(
                description
            )

        header = {
            "run_id": self.run_id,
            "description": description,
            "results": summary
        }

        with open(
            self.header_path,
            "w"
        ) as file:
            json.dump(
                json_safe(header),
                file,
                indent=4
            )

        if oracle_image is not None:
            torch.save(oracle_image.detach().cpu(), os.path.join(self.directory, "oracle_reconstruction.pt"))
            save_image(display_tensor(align_for_display(oracle_image, "oracle_reconstruction", summary)), os.path.join(self.directory, "oracle_reconstruction.png"))

        # best reconstruction
        if best_image is not None:

            image = (
                best_image
                .detach()
                .cpu()
            )

            torch.save(
                image,
                os.path.join(
                    self.directory,
                    "best_reconstruction.pt"
                )
            )

            save_image(
                display_tensor(align_for_display(image, "best_reconstruction", summary)),
                os.path.join(
                    self.directory,
                    "best_reconstruction.png"
                )
            )

        # ground truth
        if ground_truth is not None:

            truth = (
                ground_truth
                .detach()
                .cpu()
            )

            torch.save(
                truth,
                os.path.join(
                    self.directory,
                    "ground_truth.pt"
                )
            )

            save_image(
                display_tensor(truth),
                os.path.join(
                    self.directory,
                    "ground_truth.png"
                )
            )


    def close(self):
        if not self.file.closed:
            self.file.close()
