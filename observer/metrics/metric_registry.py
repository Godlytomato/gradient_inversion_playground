from observer.metrics.metric_def import metric_def
from observer.metrics.implementations.mse import mse
from observer.metrics.implementations.ssim import ssim
from observer.metrics.implementations.psnr import psnr

metric_registry = {
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