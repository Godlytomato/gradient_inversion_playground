from transformations.transform_def import transformation

# base transforms
from transformations.transformation_implimentations.zero import zero
from transformations.transformation_implimentations.noise import noise

# noise construction
from transformations.transformation_implimentations.gaussian import gaussian
from transformations.transformation_implimentations.uniform import uniform
from transformations.transformation_implimentations.laplace import laplace
from transformations.transformation_implimentations.max_svd_alignment import (
    max_svd_alignment
)
from transformations.transformation_implimentations.min_svd_alignment import (
    min_svd_alignment
)
from transformations.transformation_implimentations.clipping import (
    clipping_delta
)
from transformations.transformation_implimentations.quantization import (
    quantization_delta
)
from transformations.transformation_implimentations.reduction import (
    reduction
)


# noise scaling / thresholds
from transformations.transformation_implimentations.threasholds.preset import (
    preset
)
from transformations.transformation_implimentations.threasholds.theashold import (
    threshold
)
from transformations.transformation_implimentations.threasholds.lower_bound_threashold import (
    lower_bound_threshold
)
from transformations.transformation_implimentations.threasholds.identity import (
    identity
)
from transformations.transformation_implimentations.threasholds.relative_gradient_scale import (
    relative_gradient_scale
)
from transformations.transformation_implimentations.threasholds.max_svd_reference_scale import (
    max_svd_reference_scale
)

# --------------------
# transforms
# --------------------

transform_dict = {
    "zero": transformation(
        name="zero",
        func=zero
    ),

    "noise": transformation(
        name="noise",
        func=noise
    ),
}


# --------------------
# noise construction
# --------------------

noise_construction_dict = {
    "quantization": transformation(
        name="quantization",
        func=quantization_delta
    ),

    "gaussian": transformation(
        name="gaussian",
        func=gaussian
    ),

    "uniform": transformation(
        name="uniform",
        func=uniform
    ),

    "laplace": transformation(
        name="laplace",
        func=laplace
    ),

    "max_svd_alignment": transformation(
        name="max_svd_alignment",
        func=max_svd_alignment
    ),

    "min_svd_alignment": transformation(
        name="min_svd_alignment",
        func=min_svd_alignment
    ),
    "clipping_delta": transformation(
        name="clipping_delta",
        func=clipping_delta
    ),
    "reduction": transformation(
        name="reduction",
        func=reduction
    ),
}


# --------------------
# noise scaling
# --------------------

noise_scaling_dict = {

    "identity": transformation(
        name="identity",
        func=identity
    ),
    "preset": transformation(
        name="preset",
        func=preset
    ),

    "threshold": transformation(
        name="threshold",
        func=threshold
    ),

    "lower_bound_threshold": transformation(
        name="lower_bound_threshold",
        func=lower_bound_threshold
    ),
    "relative_gradient_scale": transformation(
        name="relative_gradient_scale",
        func=relative_gradient_scale
    ),
    "max_svd_reference_scale": transformation(
        name="max_svd_reference_scale",
        func=max_svd_reference_scale
    ),
}