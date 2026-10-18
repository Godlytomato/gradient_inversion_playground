
from dataclasses import dataclass, field
from typing import Any
import math
import copy
import torch

from run_def import resolved_observer


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


@dataclass
class observer:
    config: resolved_observer
    outcome: str = "running"

    # total iterations across all restarts
    step: int = 0

    # iterations in current restart
    attempt_step: int = 0

    current_image: Any = None
    current_metrics: dict[str, Any] = field(default_factory=dict)

    # best result across all restarts
    best_image: Any = None
    best_metrics: dict[str, Any] = field(default_factory=dict)

    # run-level measurements
    run_metrics: dict[str, Any] = field(default_factory=dict)

    ground_truth: Any = None
    run_id: str | None = None
    description: Any = None
    outcome_reason: str | None = None

    def __post_init__(self):
        self.logger = None
        if self.config.logging is not None:
            logging = self.config.logging
            if self.run_id is None:
                raise ValueError("run_id is required when logging is enabled")
            self.logger = logging.function(
                run_id=self.run_id,
                description=self.description,
                **logging.parameters
            )

    def set_run_metrics(self, metrics: dict):
        self.run_metrics.update(metrics)

    def update(self, reconstruction, metrics=None):
        self.step += 1
        self.attempt_step += 1
        self.current_image = reconstruction
        self.current_metrics = {}

        # calculate configured metrics
        for metric in self.config.metrics:
            value = metric.function(
                reconstruction,
                self.ground_truth,
                **metric.parameters
            )
            self.current_metrics[
                metric.name or metric.function.__name__
            ] = value

        # additional metrics supplied by attack
        if metrics is not None:
            self.current_metrics.update(metrics)

        self.update_best()

        if self.logger is not None:
            self.logger.log(self.step, self.current_metrics)

        # success conditions
        for policy in self.config.success_policies:
            if policy.function(self, **policy.parameters):
                self.outcome = "success"
                self.outcome_reason = policy.name or policy.function.__name__
                return

        # failure conditions
        for policy in self.config.fail_policies:
            if policy.function(self, **policy.parameters):
                self.outcome = "failure"
                self.outcome_reason = policy.name or policy.function.__name__
                return

    def should_stop(self) -> bool:
        return self.outcome != "running"

    def update_best(self):
        selection = self.config.best_selection
        if selection is None:
            return

        metric_name = selection.metric
        if metric_name not in self.current_metrics:
            raise ValueError(f"Missing selection metric: {metric_name}")

        value = self.current_metrics[metric_name]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Selection metric must be scalar")
            value = value.detach().item()

        value = float(value)
        if not math.isfinite(value):
            print("Selection metric:", metric_name)
            print("Selection value:", value)
            print("Current metrics:", self.current_metrics)
            print("Step:", self.step)

            if isinstance(self.current_image, torch.Tensor):
                print(
                    "Reconstruction finite:",
                    torch.isfinite(self.current_image).all().item()
                )

            raise ValueError(
                f"Selection metric '{metric_name}' "
                f"is not finite: {value}"
            )

        if not self.best_metrics:
            improved = True
        else:
            previous = self.best_metrics[metric_name]

            if selection.direction == "minimize":
                improved = value < previous
            elif selection.direction == "maximize":
                improved = value > previous
            else:
                raise ValueError(
                    "Best selection direction must be 'minimize' or 'maximize'"
                )

        if improved:
            self.best_image = snapshot(self.current_image)
            self.best_metrics = {
                name: snapshot(metric)
                for name, metric in self.current_metrics.items()
            }
            self.best_metrics[metric_name] = value

    def reset_attempt(self):
        self.outcome = "running"
        self.outcome_reason = None
        self.current_image = None
        self.current_metrics = {}
        self.attempt_step = 0

        if hasattr(self, "_early_fail_best"):
            del self._early_fail_best
        if hasattr(self, "_early_fail_step"):
            del self._early_fail_step

        # Preserve total steps and best reconstruction across restarts.

    def summary(self) -> dict:
        return {
            "outcome": self.outcome,
            "reason": self.outcome_reason,
            "total_iterations": self.step,
            "best_metrics": self.best_metrics.copy(),
            "run_metrics": self.run_metrics.copy(),
        }

    def close(self):
        if self.logger is not None:
            try:
                self.logger.save_summary(
                    description=self.description,
                    summary=self.summary(),
                    best_image=self.best_image,
                    ground_truth=self.ground_truth,
                )
            finally:
                self.logger.close()
                self.logger = None
