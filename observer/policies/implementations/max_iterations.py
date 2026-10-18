from observer.observer import observer


def max_iterations(
    observer: observer,
    limit: int
) -> bool:

    return observer.attempt_step >= limit