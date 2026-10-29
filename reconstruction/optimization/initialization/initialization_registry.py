from reconstruction.optimization.initialization.initialization_def import (
    initialization_def,
)

from reconstruction.optimization.initialization.implementations.random import (
    random_initialization,
)
from reconstruction.optimization.initialization.implementations.uniform import (
    uniform_initialization, uniform_sigmoid_initialization,
    smooth_uniform_initialization, smooth_uniform_sigmoid_initialization,
)


initialization_registry = {
    "smooth_uniform": initialization_def(
        name="smooth_uniform", function=smooth_uniform_initialization,
        requires={"batch_size", "input_shape", "device"},
    ),
    "smooth_uniform_sigmoid": initialization_def(
        name="smooth_uniform_sigmoid", function=smooth_uniform_sigmoid_initialization,
        requires={"batch_size", "input_shape", "device"},
    ),
    "uniform": initialization_def(
        name="uniform", function=uniform_initialization,
        requires={"batch_size", "input_shape", "device"},
    ),
    "uniform_sigmoid": initialization_def(
        name="uniform_sigmoid", function=uniform_sigmoid_initialization,
        requires={"batch_size", "input_shape", "device"},
    ),
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
