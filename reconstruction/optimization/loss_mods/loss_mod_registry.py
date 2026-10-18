from reconstruction.optimization.loss_mods.loss_mod_def import loss_mod_def

from reconstruction.optimization.loss_mods.implementations.scale import (
    scale_loss,
)
from reconstruction.optimization.loss_mods.implementations.scheduled_scale import (
    scheduled_scale,
)

loss_mod_registry = {
    "scale": loss_mod_def(
        name="scale",
        function=scale_loss,
        requires=set(),
    ),
    "scheduled_scale": loss_mod_def(
        name="scheduled_scale",
        function=scheduled_scale,
        requires=set(),
    ),
}