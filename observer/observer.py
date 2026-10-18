from dataclasses import dataclass, field
from typing import Any
import math
import copy
import torch

from run_def import resolved_observer
from observer.batch_matching import align_batch


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


@dataclass
class observer:
    config: resolved_observer
    outcome: str = "running"
    step: int = 0
    attempt_step: int = 0
    current_image: Any = None
    current_metrics: dict[str, Any] = field(default_factory=dict)
    best_image: Any = None
    best_metrics: dict[str, Any] = field(default_factory=dict)
    oracle_best_image: Any = None
    oracle_best_metrics: dict[str, Any] = field(default_factory=dict)
    run_metrics: dict[str, Any] = field(default_factory=dict)
    experiment_metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: Any = None
    run_id: str | None = None
    description: Any = None
    outcome_reason: str | None = None
    best_step: int | None = None
    oracle_best_step: int | None = None
    current_assignment: list[int] | None = None
    best_assignment: list[int] | None = None
    oracle_best_assignment: list[int] | None = None

    def __post_init__(self):
        self.logger = None
        if self.config.logging is not None:
            logging = self.config.logging
            if self.run_id is None:
                raise ValueError("run_id is required when logging is enabled")
            self.logger = logging.function(run_id=self.run_id, description=self.description, **logging.parameters)

    def set_run_metrics(self, metrics: dict):
        self.run_metrics.update(metrics)

    def update(self, reconstruction, metrics=None):
        self.step += 1
        self.attempt_step += 1
        self.current_image = reconstruction
        self.current_metrics = {}
        evaluated, self.current_assignment = align_batch(
            reconstruction, self.ground_truth, getattr(self.config, "batch_matching", "assignment")
        )
        for metric in self.config.metrics:
            value = metric.function(evaluated, self.ground_truth, **metric.parameters)
            self.current_metrics[metric.name or metric.function.__name__] = value
        if metrics is not None:
            self.current_metrics.update(metrics)
        self.update_best()
        if self.logger is not None:
            self.logger.log(self.step, self.current_metrics)
        for policy in self.config.success_policies:
            if policy.function(self, **policy.parameters):
                self.outcome = "success"
                self.outcome_reason = policy.name or policy.function.__name__
                return
        for policy in self.config.fail_policies:
            if policy.function(self, **policy.parameters):
                self.outcome = "failure"
                self.outcome_reason = policy.name or policy.function.__name__
                return

    def should_stop(self) -> bool:
        return self.outcome != "running"

    def _update_selection(self, selection, prefix):
        if selection is None:
            return
        if selection.direction not in {"minimize", "maximize"}:
            raise ValueError("Best selection direction must be 'minimize' or 'maximize'")
        metric_name = selection.metric
        if metric_name not in self.current_metrics:
            raise ValueError(f"Missing selection metric: {metric_name}")
        value = self.current_metrics[metric_name]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Selection metric must be scalar")
            value = value.detach().item()
        value = float(value)
        valid_infinity = metric_name == "psnr" and value == float('inf')
        if not math.isfinite(value) and not valid_infinity:
            raise ValueError(f"Selection metric '{metric_name}' is not finite: {value}")
        if isinstance(self.current_image, torch.Tensor) and not torch.isfinite(self.current_image).all():
            raise ValueError("Reconstruction contains nonfinite values")
        previous_metrics = getattr(self, prefix + "metrics")
        improved = not previous_metrics or (
            value < previous_metrics[metric_name] if selection.direction == "minimize"
            else value > previous_metrics[metric_name]
        )
        if improved:
            setattr(self, prefix + "image", snapshot(self.current_image))
            saved = {name: snapshot(metric) for name, metric in self.current_metrics.items()}
            saved[metric_name] = value
            setattr(self, prefix + "metrics", saved)
            setattr(self, prefix + "step", self.step)
            setattr(self, prefix + "assignment", snapshot(self.current_assignment))

    def update_best(self):
        self._update_selection(self.config.best_selection, "best_")
        self._update_selection(getattr(self.config, "oracle_selection", None), "oracle_best_")

    def reset_attempt(self):
        self.outcome = "running"
        self.outcome_reason = None
        self.current_image = None
        self.current_metrics = {}
        self.current_assignment = None
        self.attempt_step = 0
        for name in ("_early_fail_best", "_early_fail_step"):
            if hasattr(self, name):
                delattr(self, name)

    def summary(self) -> dict:
        target = getattr(self.config, "evaluation_target_mse", None)
        evaluation = {"target_mse": target, "batch_matching": getattr(self.config, "batch_matching", "assignment")}
        for name, metrics in (("practical", self.best_metrics), ("oracle", self.oracle_best_metrics)):
            value = metrics.get("mse")
            evaluation[name + "_success"] = None if target is None or value is None else bool(value <= target)
        final_outcome = self.outcome
        if target is not None and final_outcome != "error" and evaluation["practical_success"] is not None:
            final_outcome = "success" if evaluation["practical_success"] else "failure"
        selection = lambda value: None if value is None else {"metric": value.metric, "direction": value.direction}
        return {
            "outcome": final_outcome,
            "stopping_outcome": self.outcome,
            "reason": self.outcome_reason,
            "total_iterations": self.step,
            "best_metrics": snapshot(self.best_metrics),
            "best_step": self.best_step,
            "best_assignment": snapshot(self.best_assignment),
            "oracle_best_metrics": snapshot(self.oracle_best_metrics),
            "oracle_best_step": self.oracle_best_step,
            "oracle_best_assignment": snapshot(self.oracle_best_assignment),
            "evaluation": evaluation,
            "selection": {"practical": selection(self.config.best_selection),
                          "oracle": selection(getattr(self.config, "oracle_selection", None))},
            "run_metrics": snapshot(self.run_metrics),
            "experiment_metadata": snapshot(self.experiment_metadata),
            "last_metrics": snapshot(self.current_metrics),
            "image_domain": "raw_0_1",
        }

    def close(self):
        if self.logger is not None:
            try:
                self.logger.save_summary(
                    description=self.description, summary=self.summary(),
                    best_image=self.best_image, ground_truth=self.ground_truth,
                    oracle_image=self.oracle_best_image,
                )
            finally:
                self.logger.close()
                self.logger = None
