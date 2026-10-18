from database.database import (
    create_run,
    add_log,
    add_image,
    finish_run,
    save_run_metrics,
)


class database_logger:
    def __init__(
        self,
        metrics,
        run_id,
        description,
    ):
        self.metrics = metrics

        # external hash passed in by the observer/run system
        self.run_hash = run_id

        # integer primary key used internally by the database
        self.database_run_id = create_run(
            description=description
        )

    def log(
        self,
        step,
        metrics,
    ):
        selected_metrics = {
            name: metrics.get(name)
            for name in self.metrics
        }

        add_log(
            run_id=self.database_run_id,
            step=step,
            metrics=selected_metrics,
        )


    def save_summary(
        self,
        description,
        summary,
        best_image=None,
        ground_truth=None,
        oracle_image=None,
    ):
        image_hash = None

        if best_image is not None:
            image_hash = add_image(
                run_id=self.database_run_id,
                image=best_image,
                image_type="best_reconstruction",
            )

        if ground_truth is not None:
            add_image(
                run_id=self.database_run_id,
                image=ground_truth,
                image_type="ground_truth",
            )
        if oracle_image is not None:
            add_image(run_id=self.database_run_id, image=oracle_image, image_type="oracle_reconstruction")
        save_run_metrics(
            run_id=self.database_run_id,
            metrics=summary["run_metrics"],
        )
        finish_run(
            run_id=self.database_run_id,
            outcome=summary["outcome"],
            best_reconstruction_hash=image_hash,
            summary=summary,
        )


    def close(self):
        pass
