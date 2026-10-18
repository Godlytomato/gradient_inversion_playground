#!/usr/bin/env python3
"""Set success-on-MSE and failure-on-stall policies in generated run YAMLs."""
import argparse
import copy
import math
from pathlib import Path
import sys

import yaml


def configure_run(raw, *, target=0.02, patience=200, minimum_change=1e-4):
    if not isinstance(raw, dict):
        raise ValueError("Expected a run mapping")
    if not math.isfinite(target) or target < 0:
        raise ValueError("target must be finite and nonnegative")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ValueError("patience must be a positive integer")
    if not math.isfinite(minimum_change) or minimum_change <= 0:
        raise ValueError("minimum_change must be finite and positive")
    updated = copy.deepcopy(raw)
    observer = updated.get('observer')
    if observer is None:
        observer = updated['observer'] = {}
    if not isinstance(observer, dict):
        raise ValueError("observer must be a mapping")
    limits = [policy for policy in observer.get('fail_policies', [])
              if policy.get('name') == 'max_iterations']
    if not limits:
        raise ValueError("Missing max_iterations safety cap; generate a current run description first")
    observer['success_policies'] = [
        {'name': 'target_mse', 'parameters': {'threshold': target}},
    ]
    observer['fail_policies'] = [
        {'name': 'no_improvement', 'parameters': {
            'metric': 'mse', 'patience': patience, 'minimum_change': minimum_change,
        }},
        *limits,
    ]
    metrics = observer.setdefault('metrics', [])
    mse = [metric for metric in metrics if metric.get('name') == 'mse']
    if not mse:
        metrics.append({'name': 'mse', 'parameters': {}})
    elif len(mse) != 1 or mse[0].get('parameters'):
        raise ValueError("Expected a single standard mse metric without parameters")
    logging = observer.get('logging')
    if logging is not None:
        requested = logging.setdefault('metrics', [])
        if 'mse' not in requested:
            requested.append('mse')
    # Return/save the best-MSE image, so reaching the target remains success
    # even if an earlier image had a lower gradient-matching loss.
    observer['best_selection'] = {'metric': 'mse', 'direction': 'minimize'}
    observer['oracle_selection'] = {'metric': 'mse', 'direction': 'minimize'}
    observer['evaluation_target_mse'] = target
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('schedule', nargs='?', type=Path, default=Path('schedual'))
    parser.add_argument('--target', type=float, default=0.02)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--minimum-change', type=float, default=1e-4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not args.schedule.is_dir():
        parser.error(f"Schedule directory does not exist: {args.schedule}")
    paths = sorted(path for path in args.schedule.iterdir() if path.suffix in {'.yaml', '.yml'})
    if not paths:
        parser.error(f"No run YAML files found in {args.schedule}; run make_run.py first")
    # Validate every new description before changing any files.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dacite import from_dict
    from run_def import run_desc
    from run_utils.run_builder import build_run
    changes = []
    for path in paths:
        original = path.read_text()
        try:
            raw = yaml.safe_load(original)
            updated = configure_run(raw, target=args.target, patience=args.patience,
                                    minimum_change=args.minimum_change)
            build_run(from_dict(data_class=run_desc, data=updated))
        except Exception as error:
            parser.error(f"{path}: {error}")
        if updated != raw:
            changes.append((path, original, yaml.safe_dump(updated, sort_keys=False)))
    for path, original, rendered in changes:
        if not args.dry_run:
            backup = path.with_name(path.name + '.before-mse-stop.bak')
            # Preserve the first backup when this script is run again.
            if not backup.exists():
                with backup.open('x') as file:
                    file.write(original)
            temporary = path.with_name(path.name + '.mse-stop.tmp')
            with temporary.open('x') as file:
                file.write(rendered)
            temporary.replace(path)
        print(f"{'Would update' if args.dry_run else 'Updated'}: {path}")
    print(f"{len(changes)} changed, {len(paths) - len(changes)} already configured.")
    print(f"Success: MSE <= {args.target}; stall: no {args.minimum_change:g} MSE improvement "
          f"for {args.patience} iterations per restart; existing iteration caps retained.")


if __name__ == '__main__':
    main()
