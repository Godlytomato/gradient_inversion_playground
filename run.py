"""Execute YAML experiments with independent, recorded random streams."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch

from data.dataset_registry import dataset_dict
from database.database import create_run, finish_run, json_safe, run_exists
from database.hashing import run_hash
from database.setup import setup_database
from models.model_registry import model_dict
from reconciliation import reconciliation
from run_utils.grad_prod import grad_prod
from run_utils.reader import read_run
from run_utils.run_builder import build_run
from run_utils.run_handler import handle_run
from run_utils.runtime_context import runtime_context
from run_utils.seeding import seed_everything, stage_seed
from run_utils.transformer import transformer
from transformations.transform_def import transformation_context
from transformations.transformation_metrics import measure_transformation


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def execute_description(description, *, skip_existing=True, progress=None):
    progress = {} if progress is None else progress
    progress["phase"] = "validation"
    executable = build_run(description)
    if skip_existing and run_exists(run_hash(description)):
        print(f"[SKIP] {description.name} already has a completed attempt")
        return None

    seeds = {stage: stage_seed(description.experiment_seed, stage)
             for stage in ("model", "data", "defense")}
    metadata = {
        "protocol_version": description.protocol_version,
        "experiment_seed": description.experiment_seed,
        "stage_seeds": seeds,
        "dataset": description.data.dataset,
    }
    progress["metadata"] = metadata
    progress["phase"] = "model_setup"
    definition = model_dict[description.model]
    data = dataset_dict[description.data.dataset]
    seed_everything(seeds["model"])
    model = reconciliation(definition, data, definition.model_builder())
    model.eval()
    model_rng_state = torch.random.get_rng_state()
    progress["phase"] = "data_sampling"
    seed_everything(seeds["data"])
    samples = data.sample(description.data.size, description.data.classes)
    # Finish lazy parameter initialization with the model stream, even if a
    # dataset transform consumed Torch randomness while sampling.
    torch.random.set_rng_state(model_rng_state)
    for item in description.gradient_transformations:
        if item.ranking_policy.name == "soteria_sensitivity":
            layer = item.ranking_policy.parameters.get("layer_name", "auto")
            if layer != "auto" and layer not in dict(model.named_modules()):
                raise ValueError(f"Unknown Soteria layer {layer!r} on {description.model}")

    progress["phase"] = "gradient_production"
    gradients = grad_prod(model, samples)
    original = [gradient.detach().clone() for gradient in gradients]

    seed_everything(seeds["defense"])
    transformation = transformation_context(
        model=model, batch=samples,
        target_mse=description.observer.evaluation_target_mse,
        original_gradients=original,
    )
    metadata.update({
        "model_state_sha256": model_fingerprint(model),
        "model_mode": "eval",
        "model_weights": "random_initialization",
        "dataset": description.data.dataset,
        "dataset_split": getattr(data, "split", None),
        "input_space": getattr(data, "input_space", "raw_pixels"),
        "sample_indices": list(samples.indices),
        "numerical_diagnostics": transformation.numerical_diagnostics,
    })
    progress["phase"] = "gradient_transformation"
    package = transformer(gradients, description.gradient_transformations, transformation)
    attack_seed = stage_seed(description.experiment_seed, f"attack:{description.attack.base_seed}")
    executable.functions.attack.parameters["base_seed"] = attack_seed
    metadata["stage_seeds"]["attack"] = attack_seed
    context = runtime_context(
        model=model, model_definition=definition, dataset_definition=data,
        batch=samples, gradients=package,
        transformation_metrics=measure_transformation(original, package),
        experiment_metadata=metadata,
        original_gradients=original,
    )
    progress["phase"] = "attack_setup_or_execution"
    return handle_run(executable, context)


def record_error(description, error, progress, path, error_log):
    """Preserve setup errors too: they occur before an attack logger exists."""
    metadata = progress.get("metadata", {})
    already_recorded = metadata.pop("_database_error_recorded", False)
    detail = {
        "schedule_file": str(path),
        "run_name": getattr(description, "name", path.stem),
        "phase": progress.get("phase", "read_description"),
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "experiment_metadata": metadata,
        "database_recorded": bool(already_recorded),
    }
    if description is not None and not already_recorded:
        try:
            run_id = create_run(description)
            finish_run(run_id, "error", summary={
                "outcome": "error", "stopping_outcome": "error",
                "reason": f"{type(error).__name__}: {error}",
                "phase": detail["phase"], "schedule_file": str(path),
                "experiment_metadata": metadata,
            })
            detail["database_recorded"] = True
        except Exception as recording_error:
            detail["database_recording_error"] = str(recording_error)
            print(f"[LOG ERROR] Could not save database error: {recording_error}", file=sys.stderr)
    try:
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with error_log.open("a") as output:
            output.write(json.dumps(json_safe(detail), allow_nan=False) + "\n")
    except Exception as recording_error:
        print(f"[LOG ERROR] Could not append {error_log}: {recording_error}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=Path("schedual"))
    parser.add_argument("--error-log", type=Path, default=Path("results/run-errors.jsonl"))
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on a run error (debugging)")
    parser.add_argument("--strict-errors", action="store_true", help="Exit with status 1 after the batch if any runs errored")
    parser.add_argument("--tracebacks", action="store_true", help="Also print full error tracebacks to the console")
    args = parser.parse_args(argv)
    if not args.schedule.is_dir():
        parser.error(f"Schedule directory does not exist: {args.schedule}")
    setup_database()
    paths = sorted(path for path in args.schedule.iterdir()
                   if path.is_file() and path.suffix in {".yaml", ".yml"})
    completed = skipped = errors = 0
    for index, path in enumerate(paths, start=1):
        description = None
        progress = {"phase": "read_description"}
        try:
            description = read_run(str(path))
            print(f"[{index}/{len(paths)}] Starting run: {description.name}", flush=True)
            result = execute_description(description, progress=progress)
            if result is None:
                skipped += 1
            else:
                completed += 1
                print(f"[COMPLETE] {description.name}", flush=True)
        except Exception as error:
            errors += 1
            print(f"[ERROR] {path.name} ({progress['phase']}): {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            if args.tracebacks:
                traceback.print_exc()
            record_error(description, error, progress, path, args.error_log)
            if args.fail_fast:
                raise
            remaining = len(paths) - index
            print(f"[CONTINUE] Error recorded; {remaining} scheduled run(s) remaining. "
                  f"Traceback: {args.error_log}", flush=True)
    print(f"Schedule finished: {completed} completed, {skipped} skipped, {errors} errors.", flush=True)
    return 1 if args.strict_errors and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
