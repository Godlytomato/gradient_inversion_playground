import inspect
import time

import torch
import torch.nn.functional as F

from reconstruction.optimization.losses.implementations.gradient_matching_utils import masked_loss
from reconstruction.optimization.labels.label_policy_def import validate_label_state
from reconstruction.optimization.optimization_context import optimization_context
from reconstruction.optimization.optimization_state import optimization_state
from reconstruction.optimization.initialization.initialization_def import parameterized_image


def requires_closure(optimizer):
    closure = inspect.signature(optimizer.step).parameters.get("closure")
    return closure is not None and closure.default is inspect.Parameter.empty


def call_component(component, information, parameters=None, **extra):
    arguments = {name: information[name] for name in component.requires}
    arguments.update(extra)
    arguments.update(component.parameters if parameters is None else parameters)
    return component.function(**arguments)


def run_attempt(information, components, observer, restart, effort=None):
    model = information["model"]
    target_gradients = information["gradients"]
    batch_size = information["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Attack batch_size must be a positive integer")
    if effort is None:
        effort = {
            "objective_evaluations": 0,
            "gradient_evaluations": 0,
            "outer_iterations": 0,
            "restarts_started": 0,
            "_started_at": time.perf_counter(),
        }
    effort["restarts_started"] += 1

    initialization = components["initialization_policy"]
    label_policy = components["label_policy"]
    optimizer_def = components["optimizer_description"]
    loss_def = components["loss"]
    preprocessing_policy = components["preprocessing_gradients_policy"]
    missingness_policy = components["missingness_policy"]
    loss_mods = components.get("loss_mods", [])
    regularizers = components.get("regularizers", [])
    post_update_mods = components.get("post_update_mods", [])
    pre_update_mods = components.get("pre_update_mods", [])
    scheduler_def = components.get("scheduler")

    initial_reconstruction = call_component(initialization, information)
    parameterized = isinstance(initial_reconstruction, parameterized_image)
    if parameterized:
        if pre_update_mods or post_update_mods:
            raise ValueError("Parameterized images cannot use pixel gradient/update mods")
        variable = initial_reconstruction.variable
        decode = initial_reconstruction.decode
        initial_reconstruction = decode(variable)
    else:
        decode = lambda value: value
    if not isinstance(initial_reconstruction, torch.Tensor):
        raise TypeError("Initialization must return a torch.Tensor")
    if initial_reconstruction.ndim < 2 or initial_reconstruction.shape[0] != batch_size:
        raise ValueError("Initialization must return a reconstruction with the requested batch size")
    if not initial_reconstruction.is_floating_point() or not torch.isfinite(initial_reconstruction).all():
        raise ValueError("Initial reconstruction must contain finite floating-point values")
    if not parameterized:
        variable = initial_reconstruction.detach().requires_grad_(True)
    elif (not isinstance(variable, torch.Tensor) or not variable.is_floating_point()
          or not variable.is_leaf or not variable.requires_grad or not torch.isfinite(variable).all()):
        raise ValueError("Parameterized image variable must be a finite floating leaf requiring gradients")
    reconstruction = initial_reconstruction
    label_state = validate_label_state(
        call_component(label_policy, information),
        batch_size=batch_size,
        class_count=information.get("class_count"),
        device=reconstruction.device,
    )

    trainable = [variable]
    if label_state.trainable:
        trainable.append(label_state.labels)

    optimizer = optimizer_def.function(trainable, **optimizer_def.parameters)
    scheduler = (
        call_component(scheduler_def, information, optimizer=optimizer)
        if scheduler_def is not None else None
    )
    state = optimization_state(
        iteration=0,
        restart=restart,
        learning_rate=optimizer.param_groups[0]["lr"],
    )

    preprocessed_target_gradients, preprocessed_missing_mask = call_component(
        preprocessing_policy,
        information,
        target_gradients=target_gradients.grads,
        missing_mask=target_gradients.missing_mask,
    )
    closure_required = requires_closure(optimizer)

    def objective(candidate_reconstruction):
        effort["objective_evaluations"] += 1
        prediction = model(candidate_reconstruction)
        if prediction.ndim != 2 or prediction.shape[0] != batch_size:
            raise ValueError("Model must return class logits with shape [batch, classes]")
        validate_label_state(
            label_state,
            batch_size=batch_size,
            class_count=prediction.shape[1],
            device=prediction.device,
        )
        if label_state.trainable:
            probabilities = F.softmax(label_state.labels, dim=1)
            task_loss = -(probabilities * F.log_softmax(prediction, dim=1)).sum(dim=1).mean()
        else:
            task_loss = F.cross_entropy(prediction, label_state.labels)

        reconstructed_gradients = torch.autograd.grad(
            task_loss, tuple(model.parameters()), create_graph=True
        )
        # Each evaluation reconstructs one complete model-gradient observation;
        # LBFGS closure calls and FedLeak probes are included in this counter.
        effort["gradient_evaluations"] += 1

        if missingness_policy.name == "visible_only":
            adapted_reconstructed = reconstructed_gradients
            adapted_target = preprocessed_target_gradients
            adapted_mask = preprocessed_missing_mask
        else:
            adapted_reconstructed, adapted_target = call_component(
                missingness_policy,
                information,
                reconstructed_gradients=reconstructed_gradients,
                target_gradients=preprocessed_target_gradients,
                missing_mask=preprocessed_missing_mask,
            )
            adapted_mask = None
        loss_arguments = {"state": state, **loss_def.parameters}
        matching_loss = masked_loss(
            loss_function=lambda **kwargs: call_component(
                loss_def, information, parameters={}, **kwargs
            ),
            reconstructed_gradients=adapted_reconstructed,
            target_gradients=adapted_target,
            missing_mask=adapted_mask,
            **loss_arguments,
        )

        attack_loss = matching_loss
        for mod in loss_mods:
            attack_loss = call_component(
                mod, information, attack_loss=attack_loss, state=state
            )

        for regularizer in regularizers:
            parameters = regularizer.parameters.copy()
            weight = parameters.pop("weight", 1.0)
            regularizer_value = call_component(
                regularizer,
                information,
                parameters=parameters,
                reconstruction=candidate_reconstruction,
                state=state,
            )
            attack_loss = attack_loss + weight * regularizer_value

        if attack_loss.ndim != 0 or not torch.isfinite(attack_loss):
            raise FloatingPointError("Attack objective must be a finite scalar")
        return attack_loss

    context = optimization_context(objective=objective)

    while True:
        def closure():
            optimizer.zero_grad()
            attack_loss = objective(decode(variable))
            attack_loss.backward()
            for mod in pre_update_mods:
                call_component(
                    mod, information,
                    reconstruction=variable, state=state, context=context,
                )
            for parameter in trainable:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Attack produced nonfinite optimization gradients")
            return attack_loss

        if closure_required:
            attack_loss = optimizer.step(closure)
        else:
            attack_loss = closure()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        state.learning_rate = optimizer.param_groups[0]["lr"]

        for mod in post_update_mods:
            call_component(mod, information, reconstruction=variable, state=state)
        reconstruction = decode(variable)
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("Attack update produced a nonfinite reconstruction")

        state.iteration += 1
        effort["outer_iterations"] += 1
        with torch.enable_grad():
            current_objective = objective(reconstruction)
            current_loss = current_objective.detach().item()
            diagnostics = {}
            if hasattr(observer, "inspect_step"):
                diagnostics = observer.inspect_step(
                    reconstruction, variable, label_state, current_objective, state,
                )

        measurements = {
            name: value for name, value in effort.items() if not name.startswith("_")
        }
        measurements["elapsed_seconds"] = time.perf_counter() - effort["_started_at"]
        measurements["restart_index"] = restart
        if hasattr(observer, "set_run_metrics"):
            observer.set_run_metrics(measurements)
        observer.update(reconstruction, metrics={"attack_loss": current_loss, **measurements, **diagnostics})
        if observer.should_stop():
            break

    return reconstruction.detach().clone()


def optimization_attack(information, components, observer=None, restarts=1, base_seed=0):
    if observer is None:
        raise RuntimeError("Optimization attack requires an observer with a stopping policy")
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts <= 0:
        raise ValueError("Attack restarts must be a positive integer")

    last_reconstruction = None
    effort = {
        "objective_evaluations": 0,
        "gradient_evaluations": 0,
        "outer_iterations": 0,
        "restarts_started": 0,
        "_started_at": time.perf_counter(),
    }
    for restart in range(restarts):
        if restart > 0:
            observer.reset_attempt()

        seed = base_seed + restart
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        last_reconstruction = run_attempt(
            information=information,
            components=components,
            observer=observer,
            restart=restart,
            effort=effort,
        )
        if observer.outcome == "success":
            break

    if observer.best_image is not None:
        return observer.best_image.detach().clone()
    return last_reconstruction
