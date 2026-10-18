from reconstruction.optimization.pre_update.pre_update_def import (
    pre_update_def,
)

from reconstruction.optimization.pre_update.implementations.sign import (
    sign_update,
)
from reconstruction.optimization.pre_update.implementations.fedleak_gradient_regularization import (
    fedleak_gradient_regularization,
)

pre_update_registry = {

    "sign": pre_update_def(
        name="sign",
        function=sign_update,
        requires=set(),
    ),
    "fedleak_gradient_regularization": pre_update_def(
        name="fedleak_gradient_regularization",
        function=fedleak_gradient_regularization,
        requires=set(),
    ),
}