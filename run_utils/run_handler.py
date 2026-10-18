from run_def import run_requirements
from run_utils.runtime_context import runtime_context
from supplementary_information.information_registry import information_registry
from database.hashing import run_hash

def gather_information(
    requirements: run_requirements,
    context: runtime_context
) -> dict:

    information = {}
    resolving = set()

    def resolve(name):

        # Already retrieved
        if name in information:
            return information[name]

        # Detect circular dependencies
        if name in resolving:
            raise ValueError(
                f"Circular information dependency: {name}"
            )

        # Find provider
        definition = information_registry.get(name)

        if definition is None:
            raise ValueError(
                f"Unknown information requirement: {name}"
            )

        resolving.add(name)

        try:
            # Resolve dependencies first
            dependencies = {
                required: resolve(required)
                for required in definition.requires
            }

            # Retrieve information
            value = definition.provider(context, **dependencies,**definition.parameters)
            
            # Cache result
            information[name] = value

            return value

        finally:
            resolving.remove(name)

    # Gather all requested information
    all_requirements = (
        requirements.attack_information
        | requirements.observer_information
    )

    for name in sorted(all_requirements):
        resolve(name)

    return information

from run_def import run

def handle_run(
    executable_run: run,
    context: runtime_context
):

    from run_utils.seeding import seed_everything, stage_seed
    auxiliary_seed = stage_seed(executable_run.description.experiment_seed, "auxiliary")
    seed_everything(auxiliary_seed)
    context.experiment_metadata.setdefault("stage_seeds", {})["auxiliary"] = auxiliary_seed

    # Gather required runtime information
    information = gather_information(
        executable_run.requirements,
        context
    )
    attack_information = {
        name: information[name]
        for name in executable_run.requirements.attack_information
    }

    observer_information = {
        name: information[name]
        for name in executable_run.requirements.observer_information
    }

    run_id = run_hash(
    executable_run.description
    )

    # Initialize observer if configured
    active_observer = None

    if executable_run.observer is not None:
        from observer.observer import observer

        active_observer = observer(
            config=executable_run.observer,
            run_id=run_id,
            description=executable_run.description,
            ground_truth=observer_information.get("ground_truth")
        )
        active_observer.set_run_metrics(context.transformation_metrics)
        active_observer.experiment_metadata = context.experiment_metadata

    attack_error = None
    try:
        # Retrieve resolved attack
        attack = executable_run.functions.attack

        # Execute attack
        result = attack.function(
            information=attack_information,
            components=executable_run.functions.components,
            observer=active_observer,
            **attack.parameters
        )

        return result

    except Exception as error:
        attack_error = error

        # Record unexpected execution failures
        if active_observer is not None:
            active_observer.outcome = "error"
            active_observer.outcome_reason = (
                f"{type(error).__name__}: {error}"
            )

        raise

    finally:
        # Save run summary and close logging resources
        if active_observer is not None:
            database_run_id = getattr(active_observer.logger, "database_run_id", None)
            active_observer.close()
            if attack_error is not None and database_run_id is not None:
                context.experiment_metadata["_database_error_recorded"] = True
