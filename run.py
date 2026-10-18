"""Execute YAML experiments with independent, recorded random streams."""
import argparse
import hashlib
from pathlib import Path

import torch

from data.dataset_registry import dataset_dict
from database.database import run_exists
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


def execute_description(description, *, skip_existing=True):
    executable = build_run(description)
    if skip_existing and run_exists(run_hash(description)):
        print(f"[SKIP] {description.name} already has a completed attempt")
        return None

    seeds = {stage: stage_seed(description.experiment_seed, stage)
             for stage in ("model", "data", "defense")}
    definition = model_dict[description.model]
    data = dataset_dict[description.data.dataset]
    seed_everything(seeds["model"])
    model = reconciliation(definition, data, definition.model_builder())
    model.eval()
    model_rng_state = torch.random.get_rng_state()
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

    gradients = grad_prod(model, samples)
    original = [gradient.detach().clone() for gradient in gradients]

    seed_everything(seeds["defense"])
    transformation = transformation_context(
        model=model, batch=samples,
        target_mse=description.observer.evaluation_target_mse,
    )
    package = transformer(gradients, description.gradient_transformations, transformation)
    metadata = {
        "protocol_version": description.protocol_version,
        "experiment_seed": description.experiment_seed,
        "stage_seeds": seeds,
        "model_state_sha256": model_fingerprint(model),
        "model_mode": "eval",
        "model_weights": "random_initialization",
        "dataset": description.data.dataset,
        "dataset_split": getattr(data, "split", None),
        "input_space": getattr(data, "input_space", "raw_pixels"),
        "sample_indices": list(samples.indices),
        "numerical_diagnostics": transformation.numerical_diagnostics,
    }
    attack_seed = stage_seed(description.experiment_seed, f"attack:{description.attack.base_seed}")
    executable.functions.attack.parameters["base_seed"] = attack_seed
    metadata["stage_seeds"]["attack"] = attack_seed
    context = runtime_context(
        model=model, model_definition=definition, dataset_definition=data,
        batch=samples, gradients=package,
        transformation_metrics=measure_transformation(original, package),
        experiment_metadata=metadata,
    )
    return handle_run(executable, context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=Path("schedual"))
    args = parser.parse_args(argv)
    if not args.schedule.is_dir():
        parser.error(f"Schedule directory does not exist: {args.schedule}")
    setup_database()
    paths = sorted(path for path in args.schedule.iterdir() if path.suffix in {".yaml", ".yml"})
    for path in paths:
        description = read_run(str(path))
        print(f"Starting run: {description.name}")
        execute_description(description)


if __name__ == "__main__":
    main()
