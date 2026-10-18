import inspect

import torch
import torch.nn.functional as F

from reconstruction.optimization.losses.implementations.gradient_matching_utils import masked_loss
from reconstruction.optimization.labels.label_policy_def import validate_label_state
from reconstruction.optimization.optimization_context import optimization_context
from reconstruction.optimization.optimization_state import optimization_state


def requires_closure(optimizer):
    closure = inspect.signature(optimizer.step).parameters.get("closure")
    return closure is not None and closure.default is inspect.Parameter.empty


def call_component(component, information, parameters=None, **extra):
    arguments = {name: information[name] for name in component.requires}
    arguments.update(extra)
    arguments.update(component.parameters if parameters is None else parameters)
    return component.function(**arguments)


def run_attempt(information, components, observer, restart):
    model = information["model"]
    target_gradients = information["gradients"]

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

    reconstruction = call_component(initialization, information).detach().requires_grad_(True)
    label_state = validate_label_state(call_component(label_policy, information))

    trainable = [reconstruction]
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
        prediction = model(candidate_reconstruction)
        if label_state.trainable:
            probabilities = F.softmax(label_state.labels, dim=1)
            task_loss = -(probabilities * F.log_softmax(prediction, dim=1)).sum(dim=1).mean()
        else:
            task_loss = F.cross_entropy(prediction, label_state.labels)

        reconstructed_gradients = torch.autograd.grad(
            task_loss, tuple(model.parameters()), create_graph=True
        )

        if missingness_policy.name == "visible_only":
            matching_loss = masked_loss(
                loss_function=lambda **kwargs: call_component(loss_def, information, **kwargs),
                reconstructed_gradients=reconstructed_gradients,
                target_gradients=preprocessed_target_gradients,
                missing_mask=preprocessed_missing_mask,
                state=state,
            )
        else:
            adapted_reconstructed, adapted_target = call_component(
                missingness_policy,
                information,
                reconstructed_gradients=reconstructed_gradients,
                target_gradients=preprocessed_target_gradients,
                missing_mask=preprocessed_missing_mask,
            )
            matching_loss = call_component(
                loss_def,
                information,
                reconstructed_gradients=adapted_reconstructed,
                target_gradients=adapted_target,
                state=state,
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

        return attack_loss

    context = optimization_context(objective=objective)

    while True:
        def closure():
            optimizer.zero_grad()
            attack_loss = objective(reconstruction)
            attack_loss.backward()
            for mod in pre_update_mods:
                call_component(
                    mod, information,
                    reconstruction=reconstruction, state=state, context=context,
                )
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
            call_component(mod, information, reconstruction=reconstruction, state=state)

        state.iteration += 1
        with torch.enable_grad():
            current_loss = objective(reconstruction).detach().item()

        observer.update(
            reconstruction,
            metrics={"attack_loss": current_loss},
        )
        if observer.should_stop():
            break

    return reconstruction.detach().clone()


def optimization_attack(information, components, observer=None, restarts=1, base_seed=0):
    if observer is None:
        raise RuntimeError("Optimization attack requires an observer with a stopping policy")

    last_reconstruction = None
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
        )
        if observer.outcome == "success":
            break

    if observer.best_image is not None:
        return observer.best_image.detach().clone()
    return last_reconstruction