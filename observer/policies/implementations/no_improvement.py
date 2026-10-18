from observer.observer import observer


def no_improvement(
    observer: observer,
    metric: str = "mse",
    patience: int = 100,
    minimum_change: float = 1e-4,
) -> bool:
    """
    Fail when the selected metric has not improved by at least
    minimum_change for patience iterations.

    This policy assumes lower metric values are better.
    """

    current = observer.current_metrics.get(
        metric
    )

    if current is None:
        return False

    # initialize tracking
    if not hasattr(
        observer,
        "_early_fail_best"
    ):
        observer._early_fail_best = current
        observer._early_fail_step = observer.attempt_step
        return False

    # meaningful improvement
    if (
        observer._early_fail_best
        - current
        >= minimum_change
    ):
        observer._early_fail_best = current
        observer._early_fail_step = observer.attempt_step
        return False

    # stop if we have gone too long
    # without meaningful improvement
    return (
        observer.attempt_step
        - observer._early_fail_step
        >= patience
    )