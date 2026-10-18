from reconstruction.optimization.missingness.missingness_def import (
    missingness_def,
)

from reconstruction.optimization.missingness.implementations.visible_only import (
    visible_only,
)
from reconstruction.optimization.missingness.implementations.quantiezed import (
    quantized,
)
from reconstruction.optimization.missingness.implementations.clipped import (
    clipped,
)
from reconstruction.optimization.missingness.implementations.reduced import (
    reduced,
)
from reconstruction.optimization.missingness.implementations.magnitude_threshold import (
    magnitude_threshold,
)
from reconstruction.optimization.missingness.implementations.ignore_zeros import (
    ignore_zeros,
)
from reconstruction.optimization.missingness.implementations.visible_nonzero import (
    visible_nonzero,
)
from reconstruction.optimization.missingness.implementations.all import (
    all_policies,
)

missingness_registry = {

    "visible_only": missingness_def(
        name="visible_only",
        function=visible_only,
        requires=set(),
    ),
    "quantized": missingness_def(
        name="quantized",
        function=quantized,
        requires=set(),
    ),
    "clipped": missingness_def(
        name="clipped",
        function=clipped,
        requires=set(),
    ),
    "reduced": missingness_def(
        name="reduced",
        function=reduced,
        requires=set(),
    ),
    "magnitude_threshold": missingness_def(
        name="magnitude_threshold",
        function=magnitude_threshold,
        requires=set(),
    ),
    "ignore_zeros": missingness_def(
        name="ignore_zeros",
        function=ignore_zeros,
        requires=set(),
    ),
    "visible_nonzero": missingness_def(
        name="visible_nonzero",
        function=visible_nonzero,
        requires=set(),
    ),
    "all_policies": missingness_def(
        name="all_policies",
        function=all_policies,
        requires=set(),
    ),
}