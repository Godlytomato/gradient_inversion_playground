from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def soteria(
    layer_name: str = "auto",
    percent: float = 0.1,
    stabilizer: float = 0.1,
) -> list[grad_transform]:

    """Prune highest sensitivity feature columns in one Linear weight.

    ``auto`` chooses the model's last registered Linear layer. The fraction
    refers to its input features, not all model parameters. This is an explicit
    Soteria-inspired protocol, rather than a claim of paper-faithful replication.
    """

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Pruning percentage must be between 0 and 1"
        )

    if stabilizer <= 0:
        raise ValueError(
            "Stabilizer must be greater than zero"
        )

    if not layer_name:
        raise ValueError(
            "A feature layer must be specified"
        )

    return [
        grad_transform(
            name="soteria",

            ranking_policy=function_package(
                name="soteria_sensitivity",
                parameters={
                    "layer_name": layer_name,
                    "stabilizer": stabilizer,
                },
            ),

            draft_policy=function_package(
                name="feature_columns",
                parameters={
                    "percent": percent,
                },
            ),

            transform=function_package(
                name="zero",
            ),
        )
    ]


SOTERIA_PRESET = defense_preset_def(
    name="soteria",
    builder=soteria,
)
