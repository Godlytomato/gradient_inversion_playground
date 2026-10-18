from reconstruction.optimization.preprocessing_gradients.preprocessing_gradients_def import (
    preprocessing_gradients_def,
)

from reconstruction.optimization.preprocessing_gradients.implementations.none import (
    no_preprocessing,
)


preprocessing_gradients_registry = {

    "none": preprocessing_gradients_def(
        name="none",
        function=no_preprocessing,
        requires=set(),
    ),
}