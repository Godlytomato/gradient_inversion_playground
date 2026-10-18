from run_def import run_desc
from reconstruction.attack_registry import attack_registry
from reconstruction.attack_def import attack_def

from run_def import function_package, resolved_function
from reconstruction.attack_def import component_def

from run_def import resolved_observer
from observer.policies.policy_registry import policy_registry
from observer.metrics.metric_registry import metric_registry
from observer.logging.logging_registry import logging_registry

from run_def import run, run_requirements, run_functions

def resolve_attack(description: run_desc) -> attack_def:

    # Get the attack selected by the user
    attack_name = description.attack.attack_type

    # Look up its definition
    attack_definition = attack_registry.get(attack_name)

    if attack_definition is None:
        raise ValueError(
            f"Unknown attack type: {attack_name}"
        )

    return attack_definition

def resolve_component(
    description: function_package | str,
    registry: dict[str, component_def]
) -> tuple[resolved_function, set[str]]:

    # normalize strings into function packages
    if isinstance(description, str):
        description = function_package(name=description)

    # find the selected component
    definition = registry.get(description.name)

    if definition is None:
        raise ValueError(
            f"Unknown component: {description.name}"
        )

    # validate parameters
    parameters = description.parameters.copy()

    if definition.parameter_validator is not None:
        definition.parameter_validator(parameters)

    # construct executable component
    resolved = resolved_function(
        function=definition.function,
        parameters=parameters,
        name=definition.name,
        requires=set(definition.requires)
    )

    return resolved, set(definition.requires)

def resolve_attack_components(description, attack_definition):

    requirements = set(attack_definition.requires)
    components = {}

    # Go through every component supported by this attack
    for field_name, slot in attack_definition.components.items():

        selected = getattr(description.attack, field_name, None)

        # Check if the component is required
        if selected is None:
            if slot.required:
                raise ValueError(
                    f"{field_name} is required for {attack_definition.name}"
                )
            if slot.multiple:
                components[field_name] = []
            continue

        # Components that allow multiple selections
        if slot.multiple:

            if not isinstance(selected, list):
                raise TypeError(
                    f"{field_name} must be a list"
                )

            resolved_list = []

            for item in selected:
                resolved, new_requirements = resolve_component(
                    item,
                    slot.registry
                )

                resolved_list.append(resolved)
                requirements.update(new_requirements)

            components[field_name] = resolved_list

        # Components with one selection
        else:

            resolved, new_requirements = resolve_component(
                selected,
                slot.registry
            )

            components[field_name] = resolved
            requirements.update(new_requirements)

    return components, requirements

def resolve_observer(description):

    # Observer is optional
    if description.observer is None:
        return None, set()

    config = description.observer
    validate_observer(config)
    requirements = set()

    success_policies = []
    fail_policies = []
    metrics = []
    logging = None

    # Resolve success policies
    for policy in config.success_policies:
        resolved, required = resolve_component(
            policy, policy_registry
        )
        success_policies.append(resolved)
        requirements.update(required)

    # Resolve failure policies
    for policy in config.fail_policies:
        resolved, required = resolve_component(
            policy, policy_registry
        )
        fail_policies.append(resolved)
        requirements.update(required)

    # Resolve metrics
    for metric in config.metrics:
        resolved, required = resolve_component(
            metric, metric_registry
        )
        metrics.append(resolved)
        requirements.update(required)

    # Resolve logging backend
    if config.logging is not None:
        logging_config = function_package(
            name=config.logging.logging_type,
            parameters={"metrics": config.logging.metrics}
        )

        logging, required = resolve_component(
            logging_config, logging_registry
        )
        requirements.update(required)

    selection = config.best_selection

    if selection is not None:
        if selection.direction not in {"minimize", "maximize"}:
            raise ValueError("Invalid best selection direction")

        metric_names = {
            metric.name for metric in config.metrics
        }

        if selection.metric not in metric_names:
            raise ValueError(
                f"Best selection metric not configured: {selection.metric}"
            )

    return resolved_observer(
        success_policies=success_policies,
        fail_policies=fail_policies,
        metrics=metrics,
        logging=logging,
        best_selection=config.best_selection
    ), requirements

def build_run(description: run_desc) -> run:

    # Find selected attack
    attack_definition = resolve_attack(description)

    # Attack-specific validation
    if attack_definition.validator is not None:
        attack_definition.validator(description.attack)

    # Resolve attack components
    components, attack_requirements = resolve_attack_components(
        description,
        attack_definition
    )

    # Resolve observer
    observer, observer_requirements = resolve_observer(
        description
    )

    # Resolve attack itself
    attack_function = resolved_function(
        name=attack_definition.name,
        function=attack_definition.function,
        parameters={
            "restarts": description.attack.restarts,
            "base_seed": description.attack.base_seed,
        },
        requires=set(attack_definition.requires)
    )

    # Construct executable run
    return run(
        name=description.name,

        requirements=run_requirements(
            attack_information=attack_requirements,
            observer_information=observer_requirements
        ),

        functions=run_functions(
            attack=attack_function,
            components=components
        ),

        observer=observer,
        description=description
    )
def validate_observer(config):

    metrics = {
        metric.name
        for metric in config.metrics
    }

    # policies that depend on specific metrics
    for policy in config.success_policies + config.fail_policies:

        if policy.name == "target_mse" and "mse" not in metrics:
            raise ValueError(
                "target_mse requires the mse metric"
            )

    # logging can only request configured metrics
    if config.logging is not None:
        for metric in config.logging.metrics:
            if metric not in config.logging.metrics and metric != "attack_loss":
                raise ValueError(
                    f"Logging requires unconfigured metric: {metric}"
                )

    # best selection must use a configured metric
    if config.best_selection is not None:

        if config.best_selection.metric not in metrics:
            raise ValueError(
                "Best selection metric is not configured"
            )

        if config.best_selection.direction not in {
            "minimize",
            "maximize",
        }:
            raise ValueError(
                "Invalid best selection direction"
            )