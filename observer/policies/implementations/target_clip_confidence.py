"""Stop only when all labels are correct and CLIP confidence reaches the target."""


def target_clip_confidence(observer, threshold: float) -> bool:
    value = observer.current_metrics.get("clip_confidence")
    return value is not None and value >= threshold
