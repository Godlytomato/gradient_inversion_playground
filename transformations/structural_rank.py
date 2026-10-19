"""Conservative structural rank certificates for recognized gradient maps."""

import torch
from torch import nn

from models.lenet import LeNet5


_LENET_FORWARD = LeNet5.forward
_MODULE_FORWARDS = {
    nn.Conv2d: nn.Conv2d.forward,
    nn.AvgPool2d: nn.AvgPool2d.forward,
    nn.Tanh: nn.Tanh.forward,
    nn.Linear: nn.Linear.forward,
    nn.LazyLinear: nn.LazyLinear.forward,
    nn.CrossEntropyLoss: nn.CrossEntropyLoss.forward,
}
_CHILD_TYPES = {
    "conv1": (nn.Conv2d,), "conv2": (nn.Conv2d,),
    "tanh": (nn.Tanh,), "pool": (nn.AvgPool2d,),
    "fc1": (nn.Linear, nn.LazyLinear),
    "fc2": (nn.Linear,), "fc3": (nn.Linear,),
}


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def _unmodified_forward(module, expected):
    return ("forward" not in module.__dict__
            and getattr(module.forward, "__func__", None) is expected
            and not any(name in module.__dict__
                        for name in ("_call_impl", "_slow_forward", "__call__")))


def _has_hooks(module):
    names = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
             "_backward_pre_hooks")
    for child in module.modules():
        for name in names:
            for hook in getattr(child, name, {}).values():
                # LazyLinear's built-in shape initialization does not change
                # this factorization, and inspecting it must not initialize it.
                if (name == "_forward_pre_hooks" and type(child) is nn.LazyLinear
                        and getattr(hook, "__self__", None) is child
                        and getattr(hook, "__func__", None)
                        is torch.nn.modules.lazy.LazyModuleMixin._infer_parameters):
                    continue
                return True
    module_api = torch.nn.modules.module
    return any(getattr(module_api, "_global" + name, {}) for name in names)


def gradient_jacobian_rank_bound(context, images):
    """Return a proven input nullity bound for an unmodified CIFAR-style LeNet.

    Write the parameters as first-layer theta and remaining phi. After the first
    pool, h_theta(x) is the only input to the remaining network. Consequently
    g_phi(x) = q_phi(h_theta(x)), so rank(D_x g_phi) <= dim(h). The first-layer
    gradient adds at most dim(theta) independent rows. For RGB 32x32 images this
    bounds rank(D_x g) by 1536 + 456 = 1992 < 3072. Shared batch parameters add
    456 rows once, whereas the activation and input dimensions include the batch.

    This is a sufficient certificate, not a general rank detector. Unsupported
    models, losses, hooks, or altered first-layer geometry return None. The
    calculation only inspects metadata and never executes or changes the model.
    """
    model = context.model
    if (type(model) is not LeNet5
            or not _unmodified_forward(model, _LENET_FORWARD)
            or _has_hooks(model)):
        return None
    loss = getattr(context, "loss_function", None)
    if loss is not None:
        if (type(loss) is not nn.CrossEntropyLoss
                or not _unmodified_forward(loss, _MODULE_FORWARDS[nn.CrossEntropyLoss])
                or _has_hooks(loss)):
            return None
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        return None
    if any(dimension < 1 for dimension in images.shape):
        return None
    if set(model._modules) != set(_CHILD_TYPES):
        return None
    for name, types in _CHILD_TYPES.items():
        child = model._modules[name]
        if (type(child) not in types
                or not _unmodified_forward(child, _MODULE_FORWARDS[type(child)])):
            return None

    expected_names = {f"{name}.{parameter}"
                      for name in ("conv1", "conv2", "fc1", "fc2", "fc3")
                      for parameter in ("weight", "bias")}
    parameters = list(model.named_parameters(remove_duplicate=False))
    if ({name for name, _ in parameters} != expected_names
            or len({id(parameter) for _, parameter in parameters}) != len(parameters)
            or any(getattr(parameter, name, {}) for _, parameter in parameters
                   for name in ("_backward_hooks", "_post_accumulate_grad_hooks"))
            or any(True for _ in model.buffers())):
        return None
    conv = model.conv1
    pool = model.pool
    if (images.shape[1] != conv.in_channels
            or conv.out_channels != 6 or conv.groups != 1
            or _pair(conv.kernel_size) != (5, 5)
            or _pair(conv.stride) != (1, 1)
            or _pair(conv.padding) != (2, 2)
            or _pair(conv.dilation) != (1, 1)
            or conv.padding_mode != "zeros"
            or _pair(pool.kernel_size) != (2, 2)
            or _pair(pool.stride) != (2, 2)
            or _pair(pool.padding) != (0, 0)
            or pool.ceil_mode or not pool.count_include_pad
            or pool.divisor_override is not None):
        return None
    batch, _, height, width = images.shape
    pooled_height, pooled_width = height // 2, width // 2
    if min(pooled_height, pooled_width) < 1:
        return None
    input_dimension = images.numel()
    bottleneck_dimension = batch * conv.out_channels * pooled_height * pooled_width
    early_parameter_count = sum(parameter.numel() for parameter in conv.parameters())
    rank_upper_bound = min(input_dimension, bottleneck_dimension + early_parameter_count)
    if rank_upper_bound >= input_dimension:
        return None
    return {
        "method": "lenet5_first_pool_bottleneck",
        "certified": True,
        "input_dimension": input_dimension,
        "bottleneck_dimension": bottleneck_dimension,
        "early_parameter_count": early_parameter_count,
        "rank_upper_bound": rank_upper_bound,
        "nullity_lower_bound": input_dimension - rank_upper_bound,
    }
