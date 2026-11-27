import math
from run_def import grad_transform, function_package
from validation.implementations.defenses.defense_def import defense_preset_def


def indexed_svd_noise(direction: int = 1, scale: float = 0.01):
    if not isinstance(direction, int) or direction == 0:
        raise ValueError('direction must be a nonzero integer')
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('scale must be finite and nonnegative')
    return [grad_transform(
        name='indexed_svd_noise',
        ranking_policy=function_package(name='in_place'),
        draft_policy=function_package(
            name='highest_draft', parameters={'percent': 1.0},
        ),
        transform=function_package(name='noise', parameters={
            'construction': {
                'name': 'indexed_svd_alignment',
                'parameters': {'direction': direction},
            },
            'scaling': {
                'name': 'preset', 'parameters': {'scale': scale},
            },
        }),
    )]


INDEXED_SVD_NOISE_PRESET = defense_preset_def(
    name='indexed_svd_noise', builder=indexed_svd_noise,
)
