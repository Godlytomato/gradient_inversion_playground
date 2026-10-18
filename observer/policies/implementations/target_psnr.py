
from observer.observer import observer


def target_psnr(
    observer: observer,
    threshold: float,
) -> bool:

    psnr = observer.current_metrics.get(
        "psnr"
    )

    if psnr is None:
        return False

    return psnr >= threshold