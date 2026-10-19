"""Reuse a verified singular triplet for one unchanged evaluation operator.

The caller stores only successful, checked solves. This cache is local to a
transformation context, holds at most one entry per extremal mode, and never
reuses training-mode operators. Input values matter: dataset iteration can
materialize fresh tensors, so input object identities are not suitable keys.

Registered tensors and ordinary module configuration are tracked. Arbitrary
callable losses, hooks and unknown mutable attributes disable reuse because
their state cannot be inferred safely. Customized forward methods must expose
their changing state on the module: external/global state and unsafe tensor
mutation through ``.data`` cannot be detected by version-counter signatures.
"""
import copy
import hashlib
import json

import torch


_CACHE_ATTRIBUTE = "_verified_singular_triplets"
_REGISTRY_ATTRIBUTES = {
    "_parameters", "_buffers", "_modules", "_non_persistent_buffers_set",
}
_HOOK_ATTRIBUTES = {
    "_backward_pre_hooks", "_backward_hooks", "_forward_hooks",
    "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
    "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
}


def _tensor_version(tensor):
    return (
        id(tensor), tensor._version, tuple(tensor.shape), tuple(tensor.stride()),
        str(tensor.dtype), str(tensor.device), tensor.requires_grad,
    )


def _input_digest(images, labels):
    digest = hashlib.sha256()
    for tensor in (images, labels):
        value = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        # Viewing bytes also supports dtypes such as bfloat16 that NumPy does
        # not represent directly. Only the small input batch is transferred.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.digest()


def _immutable_configuration(value):
    if value is None or type(value) in (bool, int, float, str, bytes):
        return (type(value), value)
    if type(value) is tuple:
        return (tuple, tuple(_immutable_configuration(item) for item in value))
    if isinstance(value, type):
        # Torchvision ResNet keeps its norm-layer constructor as a class.
        return (type, id(value), value.__module__, value.__qualname__)
    raise TypeError("Untracked mutable or callable module configuration")


def _module_signature(root):
    modules = []
    for name, module in root.named_modules():
        attributes = []
        for attribute, value in vars(module).items():
            if attribute in _REGISTRY_ATTRIBUTES:
                continue
            if attribute in _HOOK_ATTRIBUTES:
                if value:
                    raise TypeError("Module hooks can depend on untracked state")
                continue
            attributes.append((attribute, _immutable_configuration(value)))
        modules.append((name, id(module), type(module), tuple(sorted(attributes))))
    return (
        tuple(modules),
        tuple((name, _tensor_version(value)) for name, value in root.named_parameters()),
        tuple((name, _tensor_version(value)) for name, value in root.named_buffers()),
    )


def _cache_key(context, settings, images, labels):
    modules = tuple(context.model.modules())
    if any(module.training for module in modules):
        return None
    try:
        loss = context.loss_function
        if loss is None:
            loss_signature = None
        elif isinstance(loss, torch.nn.Module):
            # The training flag is tracked, but does not itself forbid a loss
            # module: standard CrossEntropyLoss defaults to training=True.
            loss_signature = _module_signature(loss)
        else:
            return None
        return (
            id(context.model),
            _module_signature(context.model),
            loss_signature,
            json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False),
            _input_digest(images, labels),
        )
    except (RuntimeError, ValueError, TypeError):
        # Inference tensors lack version counters; lazy parameters may lack
        # shapes. Missing a cache opportunity must not prevent an actual solve.
        return None


def _cache(context):
    cache = getattr(context, _CACHE_ATTRIBUTE, None)
    if cache is None:
        cache = {}
        setattr(context, _CACHE_ATTRIBUTE, cache)
    return cache


def lookup_singular_triplet(context, which, settings, images, labels):
    """Return isolated detached tensors on a hit, restoring saved diagnostics."""
    cache = _cache(context)
    key = _cache_key(context, settings, images, labels)
    if key is None:
        cache.clear()
        return None
    entry = cache.get(which)
    if entry is None:
        return None
    if entry[0] != key:
        del cache[which]
        return None
    diagnostics = getattr(context, "numerical_diagnostics", None)
    if diagnostics is None:
        diagnostics = {}
        context.numerical_diagnostics = diagnostics
    diagnostics[which + "_svd"] = copy.deepcopy(entry[2])
    return tuple(value.detach().clone() for value in entry[1])


def store_singular_triplet(context, which, settings, images, labels, triplet, diagnostic):
    """Store a successful checked solve; replacing a mode releases its old entry."""
    cache = _cache(context)
    key = _cache_key(context, settings, images, labels)
    if key is None:
        cache.clear()
        return
    cache[which] = (
        key, tuple(value.detach().clone() for value in triplet),
        copy.deepcopy(diagnostic),
    )
