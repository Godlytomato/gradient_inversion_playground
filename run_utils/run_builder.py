import inspect
import math

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
    registry: dict[str, component_def],
    wrapper_parameters: set[str] | None = None,
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
    # Catch misspelled/unsupported parameters at the description boundary.
    try:
        supplied = {key: value for key, value in parameters.items()
                    if key not in (wrapper_parameters or set())}
        inspect.signature(definition.function).bind_partial(**supplied)
    except TypeError as error:
        raise ValueError(f"Invalid parameters for {description.name}: {error}") from error

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
                    slot.registry,
                    {"weight"} if field_name == "regularizers" else None,
                )
                if field_name == "regularizers" and "weight" in resolved.parameters:
                    weight = resolved.parameters["weight"]
                    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                        raise ValueError("Regularizer weight must be finite and nonnegative")

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

        if selection.metric not in metric_names | {"attack_loss"}:
            raise ValueError(
                f"Best selection metric not configured: {selection.metric}"
            )

    return resolved_observer(
        success_policies=success_policies,
        fail_policies=fail_policies,
        metrics=metrics,
        logging=logging,
        best_selection=config.best_selection,
        oracle_selection=config.oracle_selection,
        batch_matching=config.batch_matching,
        evaluation_target_mse=config.evaluation_target_mse,
    ), requirements

def build_run(description: run_desc) -> run:

    validate_description(description)

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

    if config.batch_matching not in {"ordered", "assignment"}:
        raise ValueError("batch_matching must be 'ordered' or 'assignment'")
    target = config.evaluation_target_mse
    if target is not None:
        if not math.isfinite(target) or target < 0 or "mse" not in metrics:
            raise ValueError("evaluation_target_mse requires a finite nonnegative target and mse metric")

    # Policies must have the data they actually evaluate.
    for policy in config.success_policies + config.fail_policies:

        dependency = {
            "target_mse": "mse", "target_psnr": "psnr", "target_ssim": "ssim",
        }.get(policy.name)
        if policy.name == "no_improvement":
            dependency = policy.parameters.get("metric", "mse")
        if dependency is not None and dependency not in metrics | {"attack_loss"}:
            raise ValueError(f"{policy.name} requires the {dependency} metric")
        if policy.name == "max_iterations":
            limit = policy.parameters.get("limit")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("max_iterations limit must be a positive integer")
        if policy.name == "no_improvement":
            patience = policy.parameters.get("patience", 100)
            change = policy.parameters.get("minimum_change", 1e-4)
            if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
                raise ValueError("no_improvement patience must be a positive integer")
            if not math.isfinite(change) or change < 0:
                raise ValueError("minimum_change must be finite and nonnegative")
        if policy.name.startswith("target_"):
            threshold = policy.parameters.get("threshold")
            if threshold is None or not math.isfinite(threshold):
                raise ValueError(f"{policy.name} requires a finite threshold")

    # logging can only request configured metrics
    if config.logging is not None:
        for metric in config.logging.metrics:
            if metric not in metrics | {"attack_loss"}:
                raise ValueError(
                    f"Logging requires unconfigured metric: {metric}"
                )

    # best selection must use a configured metric
    for selection in (config.best_selection, config.oracle_selection):
        if selection is None:
            continue

        if selection.metric not in metrics | {"attack_loss"}:
            raise ValueError(
                "Best selection metric is not configured"
            )

        if selection.direction not in {
            "minimize",
            "maximize",
        }:
            raise ValueError(
                "Invalid best selection direction"
            )


def validate_description(description):
    if isinstance(description.experiment_seed, bool) or not isinstance(description.experiment_seed, int):
        raise ValueError("experiment_seed must be an integer")
    attack = description.attack
    if isinstance(attack.restarts, bool) or not isinstance(attack.restarts, int) or attack.restarts < 1:
        raise ValueError("restarts must be a positive integer")
    if isinstance(description.data.size, bool) or not isinstance(description.data.size, int) or description.data.size < 1:
        raise ValueError("data.size must be a positive integer")
    if not description.data.classes:
        raise ValueError("data.classes must contain at least one class")
    labels = attack.label_policy
    label_name = labels if isinstance(labels, str) else labels.name
    if label_name in {"idlg", "hf_gradinv"} and description.data.size != 1:
        raise ValueError(f"{label_name} currently supports batch size 1 only")
    if attack.attack_type == "optimization":
        if description.observer is None:
            raise ValueError("optimization requires an observer with a bounded stopping policy")
        policies = description.observer.success_policies + description.observer.fail_policies
        if not any(policy.name == "max_iterations" for policy in policies):
            raise ValueError("optimization requires an explicit max_iterations safety limit")
    if any(item.name == "partial_gradient_matching" for item in attack.regularizers):
        raise ValueError("partial_gradient_matching is a matching loss; select it with attack.loss")
    validate_transformations(description.gradient_transformations)


def validate_transformations(transformations):
    from transformations.application_selection_policy_registrys import ranking_policy_dict, draft_policy_dict
    from transformations.transform_registry import transform_dict, noise_construction_dict, noise_scaling_dict

    def check(name, parameters, registry, runtime):
        if name not in registry:
            raise ValueError(f"Unknown transformation component: {name}")
        if "percent" in parameters:
            percent = parameters["percent"]
            if not isinstance(percent, (int, float)) or not math.isfinite(percent) or not 0 <= percent <= 1:
                raise ValueError("Selection percent must be finite and between 0 and 1")
        if name == "quantization" and parameters.get("mode", "bits") == "bits":
            bits = parameters.get("bits", 8)
            if isinstance(bits, bool) or not isinstance(bits, int) or bits < 2:
                raise ValueError("Signed symmetric quantization requires bits >= 2")
        arguments = dict(runtime)
        arguments.update(parameters)
        try:
            inspect.signature(registry[name].func).bind(**arguments)
        except TypeError as error:
            raise ValueError(f"Invalid parameters for {name}: {error}") from error

    sentinel = object()
    for item in transformations:
        if item.ranking_policy.name == "soteria_sensitivity" and item.draft_policy.name != "feature_columns":
            raise ValueError("Soteria requires feature_columns selection; regenerate historical Soteria YAML")
        check(item.ranking_policy.name, item.ranking_policy.parameters, ranking_policy_dict,
              {"package": sentinel, "context": sentinel})
        check(item.draft_policy.name, item.draft_policy.parameters, draft_policy_dict,
              {"scores": sentinel})
        if item.transform.name not in transform_dict:
            raise ValueError(f"Unknown transformation: {item.transform.name}")
        if item.transform.name == "zero" and item.transform.parameters:
            raise ValueError("zero transformation does not accept parameters")
        if item.transform.name == "noise":
            parts = item.transform.parameters
            for field, registry, runtime in (
                ("construction", noise_construction_dict, {"package": sentinel, "selection_masks": sentinel, "context": sentinel}),
                ("scaling", noise_scaling_dict, {"package": sentinel, "noise_tensors": sentinel, "selection_masks": sentinel, "context": sentinel}),
            ):
                part = parts.get(field)
                if not isinstance(part, dict) or "name" not in part:
                    raise ValueError(f"noise requires {field} description")
                check(part["name"], part.get("parameters", {}), registry, runtime)
