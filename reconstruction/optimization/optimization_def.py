from reconstruction.attack_def import attack_def, component_slot

from reconstruction.optimization.initialization.initialization_registry import (
    initialization_registry,
)
from reconstruction.optimization.labels.label_policy_registry import (
    label_policy_registry,
)
from reconstruction.optimization.optimizers.optimizer_registry import (
    optimizer_registry,
)
from reconstruction.optimization.losses.loss_registry import (
    loss_registry,
)
from reconstruction.optimization.regularizers.regularizer_registry import (
    regularizer_registry,
)
from reconstruction.optimization.post_update.post_update_registry import (
    post_update_registry,
)

from reconstruction.optimization.optimization_attack import (
    optimization_attack,
)
from reconstruction.optimization.loss_mods.loss_mod_registry import (
    loss_mod_registry,
)
from reconstruction.optimization.pre_update.pre_update_registry import (
    pre_update_registry,
)
from reconstruction.optimization.missingness.missingness_registry import (
    missingness_registry,
)
from reconstruction.optimization.preprocessing_gradients.preprocessing_gradients_registry import (
    preprocessing_gradients_registry,
)
from reconstruction.optimization.schedulers.scheduler_registry import (
    scheduler_registry,
)
OPTIMIZATION_ATTACK = attack_def(
    name="optimization",

    function=optimization_attack,

    requires={
        "model",
        "gradients",
        "batch_size",
    },
    

    components={
        "scheduler": component_slot(
            registry=scheduler_registry,
            required=False,
        ),
        "initialization_policy": component_slot(
            registry=initialization_registry,
            required=True,
        ),

        "label_policy": component_slot(
            registry=label_policy_registry,
            required=True,
        ),

        "optimizer_description": component_slot(
            registry=optimizer_registry,
            required=True,
        ),

        "loss": component_slot(
            registry=loss_registry,
            required=True,
        ),

        "regularizers": component_slot(
            registry=regularizer_registry,
            multiple=True,
        ),

        "post_update_mods": component_slot(
            registry=post_update_registry,
            multiple=True,
        ),
        "loss_mods": component_slot(
            registry=loss_mod_registry,
            multiple=True,
        ),
        "pre_update_mods": component_slot(
            registry=pre_update_registry,
            multiple=True,
        ),
        "missingness_policy": component_slot(
            registry=missingness_registry,
            required=True,
        ),
        "preprocessing_gradients_policy": component_slot(
            registry=preprocessing_gradients_registry,
            required=True,
        ),
    },
)