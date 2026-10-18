from dataclasses import dataclass, field
from typing import Any, Callable


# =========================================================
# Description-side definitions
# These represent what comes from the YAML/user configuration
# =========================================================


# data loading
@dataclass
class batch:
    dataset: str
    size: int
    classes: list[int]


# generic named function + parameters from configuration
@dataclass
class function_package:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


# gradient transformation description
@dataclass
class grad_transform:
    name: str
    ranking_policy: function_package
    draft_policy: function_package
    transform: function_package


# reconstruction attack description
@dataclass
class reconstruction_job:
    attack_type: str

    initialization_policy: str

    label_policy: str | function_package

    restarts: int

    optimizer_description: function_package

    loss: str | function_package

    missingness_policy: function_package | str

    preprocessing_gradients_policy: str

    loss_mods: list[function_package]

    regularizers: list[function_package]

    post_update_mods: list[function_package]

    pre_update_mods: list[function_package]

    scheduler: function_package | None

    gan_description: function_package | None = None

    base_seed: int = 0


# logging configuration
@dataclass
class logging_description:
    logging_type: str
    metrics: list[str] = field(default_factory=list)


# determines which metric controls best-image selection
@dataclass
class best_selection_description:
    metric: str
    direction: str = "minimize"


# observer configuration
@dataclass
class observer_description:
    success_policies: list[function_package] = field(default_factory=list)
    fail_policies: list[function_package] = field(default_factory=list)

    metrics: list[function_package] = field(default_factory=list)

    logging: logging_description | None = None

    best_selection: best_selection_description | None = None


# complete user-facing run description
@dataclass
class run_desc:
    name: str
    model: str
    data: batch
    gradient_transformations: list[grad_transform]
    attack: reconstruction_job

    attack_name: str | None = None
    defense_name: str | None = None

    observer: observer_description | None = None


# =========================================================
# Executable-side definitions
# These are constructed by run_builder
# =========================================================


# resolved callable ready for execution
@dataclass
class resolved_function:
    function: Callable
    parameters: dict[str, Any] = field(default_factory=dict)

    # registry name that produced this function
    name: str | None = None

    # runtime information explicitly required by this component
    requires: set[str] = field(default_factory=set)


# observer configuration after registry resolution
@dataclass
class resolved_observer:
    success_policies: list[resolved_function]
    fail_policies: list[resolved_function]
    metrics: list[resolved_function]

    logging: resolved_function | None = None
    best_selection: best_selection_description | None = None


# runtime information requirements
@dataclass
class run_requirements:

    # information allowed to be passed to the attack
    attack_information: set[str] = field(default_factory=set)

    # information used only by the observer/evaluation system
    observer_information: set[str] = field(default_factory=set)


# attack and component callables after resolution
@dataclass
class run_functions:
    attack: resolved_function

    components: dict[
        str,
        resolved_function | list[resolved_function]
    ] = field(default_factory=dict)


# executable run produced by run_builder
@dataclass
class run:
    name: str

    requirements: run_requirements
    functions: run_functions

    observer: resolved_observer | None

    # retain original configuration for logging/reproducibility
    description: run_desc