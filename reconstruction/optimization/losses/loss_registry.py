from reconstruction.optimization.losses.loss_def import (
    loss_def,
)

from reconstruction.optimization.losses.implementations.l2 import (
    l2_loss,
)
from reconstruction.optimization.losses.implementations.layerwise_l2 import (
    layerwise_l2_loss,
)

from reconstruction.optimization.losses.implementations.cosine import (
    cosine_loss,
)
from reconstruction.optimization.losses.implementations.layerwise_cosine import (
    layerwise_cosine_loss,
)
from reconstruction.optimization.losses.implementations.weighted_cosine import (
    weighted_cosine_loss,
)

from reconstruction.optimization.losses.implementations.partial_gradient_matching import (
    partial_gradient_matching,
)
from reconstruction.optimization.losses.implementations.layerwise_partial_gradient_matching import (
    layerwise_partial_gradient_matching,
)

from reconstruction.optimization.losses.implementations.stepwise_matching import (
    stepwise_matching_loss,
)


loss_registry = {

    # ---------------------------------------------------------
    # L2
    # ---------------------------------------------------------

    "l2": loss_def(
        name="l2",
        function=l2_loss,
        requires=set(),
    ),

    "layerwise_l2": loss_def(
        name="layerwise_l2",
        function=layerwise_l2_loss,
        requires=set(),
    ),

    # ---------------------------------------------------------
    # Cosine
    # ---------------------------------------------------------

    "cosine": loss_def(
        name="cosine",
        function=cosine_loss,
        requires=set(),
    ),

    "layerwise_cosine": loss_def(
        name="layerwise_cosine",
        function=layerwise_cosine_loss,
        requires=set(),
    ),

    "weighted_cosine": loss_def(
        name="weighted_cosine",
        function=weighted_cosine_loss,
        requires=set(),
    ),

    # ---------------------------------------------------------
    # Partial matching
    # ---------------------------------------------------------

    "partial_gradient_matching": loss_def(
        name="partial_gradient_matching",
        function=partial_gradient_matching,
        requires=set(),
    ),

    "layerwise_partial_gradient_matching": loss_def(
        name="layerwise_partial_gradient_matching",
        function=layerwise_partial_gradient_matching,
        requires=set(),
    ),

    # ---------------------------------------------------------
    # Stepwise
    # ---------------------------------------------------------

    "stepwise_matching": loss_def(
        name="stepwise_matching",
        function=stepwise_matching_loss,
        requires=set(),
    ),
}