from run_def import run_desc

from run_utils.run_builder import build_run


def resolve_selection(
    selected: list[str] | None,
    registry: dict,
) -> dict:

    # Empty selection means all presets
    if not selected:
        return registry.copy()

    resolved = {}

    for name in selected:

        if name not in registry:
            raise ValueError(
                f"Unknown preset: {name}"
            )

        resolved[name] = registry[name]

    return resolved


def validate_compatibility(
    attack,
    defense,
    model,
    dataset,
) -> None:

    # Validate that every preset can be constructed
    # using its default configuration.

    attack.build()
    defense.build()
    model.build()
    dataset.build()


def validate_run(
    description: run_desc,
) -> None:

    # Resolve and validate attack components,
    # observer policies, metrics, and parameters.
    build_run(description)