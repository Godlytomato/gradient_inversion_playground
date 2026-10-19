#!/usr/bin/env python3
"""Migrate minimum-SVD YAMLs to the declared minimum-nonnull protocol."""
import argparse
import copy
from pathlib import Path
import sys

import yaml

# Support both direct execution and import by the regression suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_utils.svd_settings import (  # noqa: E402
    DEFAULT_KRYLOV_CHECK_EVERY, DEFAULT_KRYLOV_MAX_ELEMENTS,
    DEFAULT_KRYLOV_MAX_ITERATIONS, DEFAULT_MIN_SVD_METHOD,
    DEFAULT_MIN_SVD_RANK_POLICY, DEFAULT_RELATIVE_RANK_TOLERANCE,
    DEFAULT_DENSE_MAX_ELEMENTS, DEFAULT_DENSE_MAX_INPUT_SIZE,
    LEGACY_BUDGET_NAMES, min_svd_parameters,
)


def configure_run(raw, *, krylov_max_iterations=DEFAULT_KRYLOV_MAX_ITERATIONS,
                  krylov_check_every=DEFAULT_KRYLOV_CHECK_EVERY,
                  krylov_max_elements=DEFAULT_KRYLOV_MAX_ELEMENTS,
                  solver_seed=0,
                  relative_rank_tolerance=DEFAULT_RELATIVE_RANK_TOLERANCE,
                  dense_max_elements=DEFAULT_DENSE_MAX_ELEMENTS,
                  dense_max_input_size=DEFAULT_DENSE_MAX_INPUT_SIZE):
    """Set one minimum-positive protocol on ranking and construction packages."""
    if not isinstance(raw, dict):
        raise ValueError("Expected a run mapping")
    solver = min_svd_parameters(
        solver_method=DEFAULT_MIN_SVD_METHOD,
        krylov_max_iterations=krylov_max_iterations,
        krylov_check_every=krylov_check_every,
        krylov_max_elements=krylov_max_elements, solver_seed=solver_seed,
        rank_policy=DEFAULT_MIN_SVD_RANK_POLICY,
        relative_rank_tolerance=relative_rank_tolerance,
        dense_max_elements=dense_max_elements, dense_max_input_size=dense_max_input_size,
    )
    updated = copy.deepcopy(raw)
    transformations = updated.get("gradient_transformations", [])
    if not isinstance(transformations, list):
        raise ValueError("gradient_transformations must be a list")

    def migrate(package):
        parameters = package.setdefault("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("Minimum-SVD parameters must be a mapping")
        for name in LEGACY_BUDGET_NAMES:
            parameters.pop(name, None)
        parameters.update(solver)

    for transformation in transformations:
        if not isinstance(transformation, dict):
            raise ValueError("Each gradient transformation must be a mapping")
        ranking = transformation.get("ranking_policy")
        if isinstance(ranking, dict) and ranking.get("name") == "min_svd":
            migrate(ranking)
        transform = transformation.get("transform")
        if not isinstance(transform, dict) or transform.get("name") != "noise":
            continue
        parameters = transform.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("Noise parameters must be a mapping")
        construction = parameters.get("construction")
        if isinstance(construction, dict) and construction.get("name") == "min_svd_alignment":
            migrate(construction)
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", nargs="?", type=Path, default=Path("schedual"))
    parser.add_argument("--krylov-max-iterations", type=int, default=DEFAULT_KRYLOV_MAX_ITERATIONS)
    parser.add_argument("--krylov-check-every", type=int, default=DEFAULT_KRYLOV_CHECK_EVERY)
    parser.add_argument("--krylov-max-elements", type=int, default=DEFAULT_KRYLOV_MAX_ELEMENTS)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--relative-rank-tolerance", type=float,
                        default=DEFAULT_RELATIVE_RANK_TOLERANCE)
    parser.add_argument("--dense-max-elements", type=int, default=DEFAULT_DENSE_MAX_ELEMENTS)
    parser.add_argument("--dense-max-input-size", type=int, default=DEFAULT_DENSE_MAX_INPUT_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.schedule.is_dir():
        parser.error(f"Schedule directory does not exist: {args.schedule}")
    paths = sorted(path for path in args.schedule.iterdir() if path.suffix in {".yaml", ".yml"})
    if not paths:
        parser.error(f"No run YAML files found in {args.schedule}; run make_run.py first")

    from dacite import from_dict
    from run_def import run_desc
    from run_utils.run_builder import build_run

    changes = []
    # Prevalidate the entire directory before writing any schedule or backup.
    for path in paths:
        original = path.read_text()
        try:
            raw = yaml.safe_load(original)
            updated = configure_run(
                raw, krylov_max_iterations=args.krylov_max_iterations,
                krylov_check_every=args.krylov_check_every,
                krylov_max_elements=args.krylov_max_elements,
                solver_seed=args.solver_seed,
                relative_rank_tolerance=args.relative_rank_tolerance,
                dense_max_elements=args.dense_max_elements,
                dense_max_input_size=args.dense_max_input_size,
            )
            build_run(from_dict(data_class=run_desc, data=copy.deepcopy(updated)))
        except Exception as error:
            parser.error(f"{path}: {error}")
        if updated != raw:
            changes.append((path, original, yaml.safe_dump(updated, sort_keys=False)))
    for path, original, rendered in changes:
        if not args.dry_run:
            backup = path.with_name(path.name + ".before-min-svd-solver.bak")
            if not backup.exists():
                with backup.open("x") as file:
                    file.write(original)
            temporary = path.with_name(path.name + ".min-svd-solver.tmp")
            with temporary.open("x") as file:
                file.write(rendered)
            temporary.replace(path)
        print(f"{'Would update' if args.dry_run else 'Updated'}: {path}")
    print(f"{len(changes)} changed, {len(paths) - len(changes)} already configured or unaffected.")
    print(f"Minimum-SVD protocol: {DEFAULT_MIN_SVD_METHOD}, rank_policy=smallest_nonzero, "
          f"relative_rank_tolerance={args.relative_rank_tolerance:g}; "
          "existing attack, mask, noise, and observer settings retained.")


if __name__ == "__main__":
    main()
