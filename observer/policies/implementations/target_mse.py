from observer.observer import observer


def target_mse(
    observer: observer,
    threshold: float
) -> bool:

    mse = observer.current_metrics.get("mse")

    if mse is None:
        return False

    return mse <= threshold