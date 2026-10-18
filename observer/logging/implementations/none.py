class none_logger:
    """
    Logger used for validation and smoke tests.

    It implements the normal logger interface but does not
    write anything to disk or to the experiment database.
    """

    def __init__(
        self,
        metrics,
        run_id,
        description=None,
    ):
        self.metrics = metrics
        self.run_id = run_id


    def log(
        self,
        step,
        metrics,
    ):
        pass


    def save_summary(
        self,
        description,
        summary,
        best_image=None,
        ground_truth=None,
    ):
        pass


    def close(self):
        pass