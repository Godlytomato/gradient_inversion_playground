# observer/policies/implementations/target_ssim.py

from observer.observer import observer


def target_ssim(
    observer: observer,
    threshold: float,
) -> bool:

    ssim = observer.current_metrics.get(
        "ssim"
    )

    if ssim is None:
        return False

    return ssim >= threshold