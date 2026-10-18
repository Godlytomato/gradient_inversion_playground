from observer.policies.policy_def import policy_def

from observer.policies.implementations.max_iterations import (
    max_iterations,
)

from observer.policies.implementations.target_mse import (
    target_mse,
)

from observer.policies.implementations.target_psnr import (
    target_psnr,
)

from observer.policies.implementations.target_ssim import (
    target_ssim,
)
from observer.policies.implementations.no_improvement import (
    no_improvement,
)


policy_registry = {

    "max_iterations": policy_def(
        name="max_iterations",
        function=max_iterations,
        requires=set(),
    ),

    "target_mse": policy_def(
        name="target_mse",
        function=target_mse,
        requires={"ground_truth"},
    ),

    "target_psnr": policy_def(
        name="target_psnr",
        function=target_psnr,
        requires={"ground_truth"},
    ),

    "target_ssim": policy_def(
        name="target_ssim",
        function=target_ssim,
        requires={"ground_truth"},
    ),
    "no_improvement": policy_def(
        name="no_improvement",
        function=no_improvement,
        requires={"ground_truth"},
    ),
}