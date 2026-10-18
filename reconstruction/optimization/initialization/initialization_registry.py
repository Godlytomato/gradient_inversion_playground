from reconstruction.optimization.initialization.initialization_def import (
    initialization_def,
)

from reconstruction.optimization.initialization.implementations.random import (
    random_initialization,
)


initialization_registry = {
    "random": initialization_def(
        name="random",
        function=random_initialization,
        requires={
            "batch_size",
            "input_shape",
            "device",
        },
    ),
}