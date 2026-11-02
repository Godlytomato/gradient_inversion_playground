from observer.metrics.metric_def import metric_def
from observer.metrics.implementations.mse import mse
from observer.metrics.implementations.ssim import ssim
from observer.metrics.implementations.psnr import psnr
from observer.metrics.implementations.clip_confidence import clip_confidence, validate_clip_parameters

metric_registry = {
    "clip_confidence": metric_def(
        name="clip_confidence",
        function=clip_confidence,
        requires={"ground_truth", "labels", "data_def"},
        parameter_validator=validate_clip_parameters,
    ),
    "mse": metric_def(
        name="mse",
        function=mse,
        requires={"ground_truth"}
    ),
    "ssim": metric_def(
        name="ssim",
        function=ssim,
        requires={"ground_truth"}
    ),
    "psnr": metric_def(
        name="psnr",
        function=psnr,
        requires={"ground_truth"}
    )
}
