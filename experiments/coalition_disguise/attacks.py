"""Registered attacks with explicit, experiment-local calibration overrides."""
import csv
import hashlib
import inspect
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import torch
from torchvision.utils import save_image

from .common import digest, model_digest, pixel_metrics, tensor_digest, unpack, write_json
from observer.observer import observer
from run_def import (run_desc, batch, observer_description, function_package,
                     best_selection_description)
from run_utils.run_builder import build_run
from validation.registries.registry_builder import build_registries


def source_fingerprint():
    """Version execution separately from the immutable decomposition inputs."""
    root = Path(__file__).resolve().parents[2]
    files = set()
    for directory in ('experiments/coalition_disguise', 'reconstruction', 'observer',
                      'models', 'validation/implementations/attacks'):
        files.update((root / directory).rglob('*.py'))
    files.update([root / 'run_def.py', root / 'run_utils/run_builder.py'])
    return {str(p.relative_to(root)): digest(p) for p in sorted(files)}


def build_attack(name, args):
    preset = build_registries()['attacks'][name]
    job = preset.build()
    job.base_seed = args.seed
    profile = getattr(args, 'attack_profile', 'calibrated')
    if profile == 'calibrated':
        # Opt-in to bounded coordinates; never project LBFGS iterates after a
        # step, which would invalidate its curvature history/line search.
        if name in {'idlg', 'dlg'}:
            tolerances = (getattr(args, 'lbfgs_tolerance_grad', 1e-12),
                          getattr(args, 'lbfgs_tolerance_change', 1e-15))
            if any(not math.isfinite(value) or value <= 0 for value in tolerances):
                raise ValueError('LBFGS tolerances must be finite and positive')
            job.initialization_policy = 'smooth_uniform_sigmoid'
            job.restarts = 4
            job.optimizer_description.parameters.update(
                tolerance_grad=getattr(args, 'lbfgs_tolerance_grad', 1e-12),
                tolerance_change=getattr(args, 'lbfgs_tolerance_change', 1e-15))
        else:
            job.initialization_policy = 'uniform'
            job.optimizer_description.parameters['lr'] = .01
    elif profile != 'preset':
        raise ValueError('Unknown attack profile')
    if getattr(args, 'initialization', None) is not None:
        job.initialization_policy = args.initialization
    if (job.initialization_policy in {'uniform_sigmoid', 'smooth_uniform_sigmoid'}
            and (job.pre_update_mods or job.post_update_mods)):
        raise ValueError('Sigmoid initialization cannot use pixel gradient/update mods; choose a pixel initializer for this preset')
    if args.restarts is not None:
        job.restarts = args.restarts
    if args.lr is not None:
        job.optimizer_description.parameters['lr'] = args.lr
    if args.max_iterations is not None and args.max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    if args.minimum_change <= 0:
        raise ValueError('minimum_change must be positive; zero prevents plateau stopping')
    config = observer_description(
        fail_policies=[function_package('no_improvement', dict(metric='attack_loss',
                            patience=args.patience, minimum_change=args.minimum_change)),
                       function_package('max_iterations', dict(limit=args.max_iterations or preset.max_iterations))],
        best_selection=best_selection_description('attack_loss'))
    description = run_desc(name='coalition_disguise_' + name, model=getattr(args, 'model', 'lenet5'),
                           data=batch(getattr(args, 'dataset', 'cifar10'), 1, list(range(getattr(args, 'class_count', 10)))),
                           gradient_transformations=[], attack=job, attack_name=name,
                           observer=config, experiment_seed=args.seed)
    executable = build_run(description)
    # Compatibility adapter is confined to this explicitly stall-only experiment.
    if args.max_iterations is None:
        executable.observer.fail_policies = [p for p in executable.observer.fail_policies if p.name != 'max_iterations']
        description.observer.fail_policies = [p for p in description.observer.fail_policies if p.name != 'max_iterations']
    if executable.requirements.attack_information - {
            'model', 'gradients', 'batch_size', 'input_shape', 'class_count', 'device', 'labels'}:
        raise ValueError('Preset requests information outside the attacker allowlist')
    return executable


def execution_config(executable, args):
    optimizer = executable.functions.components['optimizer_description']
    effective = inspect.signature(optimizer.function).bind_partial([], **optimizer.parameters)
    effective.apply_defaults()
    return dict(protocol='reconstruction_v2', profile=getattr(args, 'attack_profile', 'calibrated'),
                description=asdict(executable.description), source_sha256=source_fingerprint(),
                optimizer_effective_parameters={k: v for k, v in effective.arguments.items() if k != 'parameters'},
                diagnostics_interval=getattr(args, 'diagnostics_interval', 20),
                torch_version=str(torch.__version__), python_version=platform.python_version(), device='cpu')


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


class TraceObserver(observer):
    """Attacker-only diagnostics; no private/public image is accepted here."""
    columns = ('step', 'restart', 'attack_loss', 'pixel_min', 'pixel_max',
               'outside_pixel_fraction', 'pixel_step_max', 'image_gradient_linf',
               'variable_gradient_linf', 'selected_label')

    def __init__(self, config, path, interval=20):
        super().__init__(config=config)
        if interval < 1:
            raise ValueError('diagnostics_interval must be positive')
        self.interval = interval
        self.stream = path.open('w')
        self.writer = csv.DictWriter(self.stream, fieldnames=self.columns)
        self.writer.writeheader()
        self.attempt_reasons = []
        self.attempt_diagnostics = []
        self.previous = None
        self.previous_restart = None

    def inspect_step(self, image, variable, labels, objective, state):
        data = pixel_metrics(image.detach())
        previous = self.previous if self.previous_restart == state.restart else None
        data['pixel_step_max'] = None if previous is None else float((image.detach() - previous).abs().max())
        self.previous = image.detach().clone()
        self.previous_restart = state.restart
        if labels.trainable:
            probabilities = labels.labels.detach().softmax(1)
            data['label_probabilities'] = probabilities.cpu().tolist()
            data['selected_label'] = int(probabilities.argmax(1).item())
        else:
            data['selected_label'] = int(labels.labels.item())
        data.update(image_gradient_linf=None, variable_gradient_linf=None)
        if state.iteration == 1 or state.iteration % self.interval == 0:
            inputs = [image] if image is variable else [image, variable]
            grads = torch.autograd.grad(objective, inputs, allow_unused=True)
            norms = [None if g is None else float(g.abs().max()) for g in grads]
            data['image_gradient_linf'] = norms[0]
            data['variable_gradient_linf'] = norms[-1]
        return data

    def update(self, reconstruction, metrics=None):
        super().update(reconstruction, metrics)
        row = dict(step=self.step, restart=metrics['restart_index'])
        row.update({k: metrics.get(k) for k in self.columns if k not in row})
        self.writer.writerow(row)
        if self.step % 100 == 0:
            self.stream.flush()
            print(f"  restart {metrics['restart_index']} step {self.attempt_step}: objective={metrics['attack_loss']:.6g}", flush=True)
        if self.should_stop():
            self.attempt_reasons.append(self.outcome_reason)
            self.attempt_diagnostics.append(dict(restart=metrics['restart_index'],
                reason=self.outcome_reason, final_objective=metrics['attack_loss'],
                iterations=self.attempt_step,
                last_pixel_step_max=metrics.get('pixel_step_max')))

    def close(self):
        self.stream.close()
        super().close()


def attack(model, observation, label, name, args, destination, *, input_shape=None, class_count=None):
    input_shape = tuple(input_shape or getattr(args, 'input_shape', (3, 32, 32)))
    class_count = class_count or getattr(args, 'class_count', 10)
    destination.mkdir(exist_ok=True)
    executable = build_attack(name, args)
    configuration = execution_config(executable, args)
    identity = dict(config=configuration, observation_sha256=tensor_digest(observation),
                    model_sha256=model_digest(model), input_shape=list(input_shape), class_count=class_count,
                    known_label_sha256=tensor_digest(label) if 'labels' in executable.requirements.attack_information else None)
    job_path = destination / 'job.json'
    if job_path.exists():
        if json.loads(job_path.read_text()) != identity:
            raise ValueError('Job inputs/configuration differ; choose a new configuration directory')
    elif (destination / 'summary.json').exists():
        raise ValueError('Historical job has no input identity; preserve it and use a new configuration directory')
    else:
        write_json(job_path, identity)
    summary_path = destination / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if digest(destination / 'reconstruction.pt') != summary['reconstruction_sha256']:
            raise ValueError('Saved reconstruction changed; refusing resume')
        print(f'  reusing completed {destination}', flush=True)
        return summary, torch.load(destination / 'reconstruction.pt', weights_only=True)
    write_json(destination / 'attack_config.json', configuration)
    from transformations.transform_def import grad_package
    information = dict(model=model, gradients=grad_package(unpack(observation, model)),
                       batch_size=1, input_shape=input_shape, class_count=class_count,
                       device=next(model.parameters()).device, labels=label)
    information = {key: information[key] for key in executable.requirements.attack_information}
    active = TraceObserver(executable.observer, destination / 'trajectory.csv',
                           interval=getattr(args, 'diagnostics_interval', 20))
    try:
        image = executable.functions.attack.function(information=information,
                    components=executable.functions.components, observer=active,
                    **executable.functions.attack.parameters)
        summary = active.summary()
        summary.update(attempt_stopping_reasons=active.attempt_reasons,
                       attempt_diagnostics=active.attempt_diagnostics,
                       input_identity={k: v for k, v in identity.items() if k != 'config'},
                       execution_protocol='reconstruction_v2', preset=name,
                       profile=getattr(args, 'attack_profile', 'calibrated'), configuration=config_hash(configuration))
    finally:
        active.close()
    torch.save(image, destination / 'reconstruction.pt')
    save_image(image.clamp(0, 1), destination / 'reconstruction.png')
    summary['reconstruction_sha256'] = digest(destination / 'reconstruction.pt')
    # Save attack completion independently of evaluation, so an interrupted
    # evaluator cannot force an expensive completed attack to rerun.
    write_json(summary_path, summary)
    return summary, image
