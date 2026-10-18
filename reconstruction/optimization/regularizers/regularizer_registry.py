from reconstruction.optimization.regularizers.regularizer_def import (
    regularizer_def,
)

from reconstruction.optimization.regularizers.implementations.tv import (
    tv_regularizer,
)
from reconstruction.optimization.regularizers.implementations.bn_stats import (
    bn_stats_regularizer,
)
from reconstruction.optimization.regularizers.implementations.auxiliary_prior import (
    auxiliary_prior_regularizer,
)
from reconstruction.optimization.regularizers.implementations.partial_gradient_matching import (
    partial_gradient_matching,
)


def reject_gradient_matching_regularizer(parameters):
    raise ValueError(
        "partial_gradient_matching is a gradient loss, not an image regularizer; "
        "select it through attack.loss instead"
    )

regularizer_registry = {
    "tv": regularizer_def(
        name="tv",
        function=tv_regularizer,
        requires=set(),
    ),
    "bn_stats": regularizer_def(
        name="bn_stats",
        function=bn_stats_regularizer,
        requires={"model"},
    ),
    "auxiliary_prior": regularizer_def(
        name="auxiliary_prior",
        function=auxiliary_prior_regularizer,
        requires={"auxiliary_images"},
    ),
    "partial_gradient_matching": regularizer_def(
        name="partial_gradient_matching",
        function=partial_gradient_matching,
        requires=set(),
        parameter_validator=reject_gradient_matching_regularizer,
    ),
}
