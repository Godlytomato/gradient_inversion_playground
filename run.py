
from pathlib import Path

from run_utils.reader import read_run
from run_utils.run_builder import build_run
from run_utils.run_handler import handle_run
from run_utils.runtime_context import runtime_context
from run_utils.grad_prod import grad_prod
from run_utils.transformer import transformer

from models.model_registry import model_dict
from data.dataset_registry import dataset_dict
from reconciliation import reconciliation

from transformations.transform_def import transformation_context
from transformations.transformation_metrics import measure_transformation

from database.hashing import run_hash
from database.database import run_exists


# folder containing run yaml files
sched_folder = Path("schedual")


# ============================================================
# build schedule
# ============================================================

schedual = []

for path in sorted(sched_folder.iterdir()):
    if path.suffix not in {".yaml", ".yml"}:
        continue

    print(f"Reading run: {path}")
    description = read_run(str(path))
    description_hash = run_hash(description)

    if run_exists(description_hash):
        print(f"[SKIP] {description.name} already exists in database")
        continue

    schedual.append(description)


# ============================================================
# execute each run description
# ============================================================

for description in schedual:
    print(f"\nStarting run: {description.name}")

    # Build model
    model_def = model_dict[description.model]
    model = model_def.model_builder()

    # Load dataset
    data_def = dataset_dict[description.data.dataset]
    sample_data = data_def.sample(
        sample_size=description.data.size,
        sampled_classes=description.data.classes,
    )

    # Reconcile model and dataset
    model = reconciliation(
        model_def=model_def,
        data=data_def,
        model=model,
    )
    model.eval()

    # Produce original gradients
    gradients = grad_prod(model=model, batch=sample_data)

    # Gradient transformation context
    transformation_info = transformation_context(
        model=model,
        batch=sample_data,
    )

    # Save original gradients before in-place transformations
    original_gradients = [g.detach().clone() for g in gradients]

    # Apply transformations exactly once
    gradient_package = transformer(
        grads=gradients,
        transformation_list=description.gradient_transformations,
        context=transformation_info,
    )

    # Measure total transformation magnitude
    transformation_metrics = measure_transformation(
        original_gradients,
        gradient_package,
    )

    # Build runtime context
    context = runtime_context(
        model=model,
        model_definition=model_def,
        dataset_definition=data_def,
        batch=sample_data,
        gradients=gradient_package,
        transformation_metrics=transformation_metrics,
    )

    # Resolve run description into executable run
    executable_run = build_run(description)

    # Execute attack and record results
    reconstruction = handle_run(executable_run, context)

    print(f"Finished run: {description.name}")
