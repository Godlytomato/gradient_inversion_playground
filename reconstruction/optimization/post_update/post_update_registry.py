from reconstruction.optimization.post_update.post_update_def import (
    post_update_def,
)

from reconstruction.optimization.post_update.implementations.clamp import (
    clamp_update,
)
from reconstruction.optimization.post_update.implementations.normalized_clamp import (
    normalized_clamp,
)

post_update_registry = {
    "clamp": post_update_def(
        name="clamp",
        function=clamp_update,
        requires=set(),
    ),
    "normalized_clamp": post_update_def(
        name="normalized_clamp",
        function=normalized_clamp,
        requires={"normalization"},
    ),
}