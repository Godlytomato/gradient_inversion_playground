from reconstruction.optimization.optimizers.optimizer_def import optimizer_def

from reconstruction.optimization.optimizers.implementations.adam import (
    adam_optimizer,
)

from reconstruction.optimization.optimizers.implementations.lbfgs import (
    lbfgs_optimizer,
)
from reconstruction.optimization.optimizers.implementations.direct_gradient_optimizer import (
    direct_gradient_optimizer,
)

optimizer_registry = {

    "adam": optimizer_def(
        name="adam",
        function=adam_optimizer,
        requires=set(),
    ),

    "lbfgs": optimizer_def(
        name="lbfgs",
        function=lbfgs_optimizer,
        requires=set(),
    ),
    "direct_gradient": optimizer_def(
        name="direct_gradient",
        function=direct_gradient_optimizer,
        requires=set(),
    ),
}