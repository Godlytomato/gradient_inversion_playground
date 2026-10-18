import unittest

import torch
from torch import nn

from reconstruction.optimization.labels.implementations.hf_gradinv_labls import hf_gradinv_labels
from reconstruction.optimization.labels.implementations.idlg import idlg_labels
from reconstruction.optimization.labels.label_policy_def import label_state, validate_label_state
from reconstruction.optimization.loss_mods.implementations.scale import scale_loss
from reconstruction.optimization.losses.implementations.gradient_matching_utils import masked_loss
from reconstruction.optimization.losses.implementations.l2 import l2_loss
from reconstruction.optimization.losses.implementations.weighted_cosine import weighted_cosine_loss
from reconstruction.optimization.missingness.implementations.visible_only import visible_only
from reconstruction.optimization.optimization_attack import optimization_attack, run_attempt
from run_def import batch, function_package, observer_description, resolved_function, run_desc
from run_utils.run_builder import build_run
from transformations.transform_def import grad_package
from validation.implementations.attacks.dlg import dlg
from validation.registries.registry_builder import build_registries


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2), nn.Tanh(), nn.Flatten()
        )
        self.classifier = nn.Linear(32, 3)

    def forward(self, images):
        return self.classifier(self.features(images))


class OneStepObserver:
    def __init__(self):
        self.outcome = "running"
        self.best_image = None
        self.metrics = {}
        self.run_metrics = {}

    def update(self, reconstruction, metrics):
        self.metrics = metrics
        self.outcome = "failure"

    def should_stop(self):
        return self.outcome != "running"

    def reset_attempt(self):
        self.outcome = "running"

    def set_run_metrics(self, metrics):
        self.run_metrics.update(metrics)


def attack_information():
    model = TinyClassifier().eval()
    images = torch.rand(1, 1, 4, 4)
    labels = torch.tensor([1])
    gradients = torch.autograd.grad(
        nn.functional.cross_entropy(model(images), labels), tuple(model.parameters())
    )
    return {
        "model": model,
        "gradients": grad_package(grads=list(gradients)),
        "labels": labels,
        "batch_size": 1,
        "input_shape": (1, 4, 4),
        "device": "cpu",
        "class_count": 3,
        "auxiliary_images": torch.rand(3, 1, 4, 4),
        "normalization": {"mean": [.4914], "std": [.247]},
    }


def attack_components(job):
    description = run_desc(
        name="synthetic", model="tiny", data=batch("synthetic", 1, [0, 1, 2]),
        gradient_transformations=[], attack=job,
        observer=observer_description(
            fail_policies=[function_package("max_iterations", {"limit": 1})]
        ),
    )
    return build_run(description).functions.components


class AttackRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_dlg_and_idlg_preserve_observed_zero_constraints(self):
        registries = build_registries()
        for name in ("dlg", "idlg"):
            self.assertEqual(registries["attacks"][name].build().missingness_policy, "visible_only")
        reconstructed, target = visible_only([torch.tensor([1., 1.])], [torch.tensor([0., 1.])], None)
        self.assertEqual(l2_loss(reconstructed, target).item(), 1.)

    def test_scale_modifier_executes_through_real_objective(self):
        self.assertEqual(scale_loss(attack_loss=torch.tensor(3.), scale=2.).item(), 6.)
        job = dlg()
        job.loss_mods = [function_package("scale", {"scale": 2.})]
        job.optimizer_description = function_package("adam", {"lr": .01})
        observer = OneStepObserver()
        result = run_attempt(attack_information(), attack_components(job), observer, 0)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(torch.tensor(observer.metrics["attack_loss"])))

    def test_unsupported_gradient_regularizer_is_rejected_during_build(self):
        job = dlg()
        job.regularizers = [function_package("partial_gradient_matching", {"weight": .1})]
        with self.assertRaisesRegex(ValueError, "attack.loss"):
            attack_components(job)

    def test_masked_weighted_cosine_preserves_original_layer_indices(self):
        reconstructed = [torch.tensor([1., 0.]), torch.tensor([1., 0.]), torch.tensor([0., 1.])]
        target = [torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([1., 0.])]
        masks = [torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)]
        loss = masked_loss(
            weighted_cosine_loss, reconstructed, target, masks,
            layer_weights=[1., 100., 3.],
        )
        self.assertAlmostEqual(loss.item(), .75)

    def test_masked_weighted_cosine_runs_with_configured_loss_parameters(self):
        job = dlg()
        information = attack_information()
        parameter_count = len(information["gradients"].grads)
        job.loss = function_package("weighted_cosine", {"layer_weights": list(range(1, parameter_count + 1))})
        job.optimizer_description = function_package("adam", {"lr": .01})
        masks = [torch.zeros_like(g, dtype=torch.bool) for g in information["gradients"].grads]
        masks[1].fill_(True)
        information["gradients"].missing_mask = masks
        result = run_attempt(information, attack_components(job), OneStepObserver(), 0)
        self.assertTrue(torch.isfinite(result).all())

    def test_zero_filtering_keeps_weighted_loss_metadata_aligned(self):
        job = dlg()
        information = attack_information()
        parameter_count = len(information["gradients"].grads)
        job.loss = function_package("weighted_cosine", {"layer_weights": list(range(1, parameter_count + 1))})
        job.missingness_policy = "visible_nonzero"
        job.optimizer_description = function_package("adam", {"lr": .01})
        information["gradients"].grads[1].zero_()
        result = run_attempt(information, attack_components(job), OneStepObserver(), 0)
        self.assertTrue(torch.isfinite(result).all())

    def test_idlg_and_hf_reject_unjustified_multi_image_label_inference(self):
        information = attack_information()
        with self.assertRaisesRegex(ValueError, "batch_size=1 only"):
            idlg_labels(information["gradients"], batch_size=2)
        with self.assertRaisesRegex(ValueError, "batch_size=1 only"):
            hf_gradinv_labels(information["model"], information["gradients"], information["auxiliary_images"], batch_size=2)

    def test_invalid_label_shapes_and_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Label batch size"):
            validate_label_state(label_state(torch.tensor([1])), batch_size=2)
        with self.assertRaisesRegex(ValueError, "output range"):
            validate_label_state(label_state(torch.tensor([3])), batch_size=1, class_count=3)
        with self.assertRaisesRegex(ValueError, "floating logits"):
            validate_label_state(label_state(torch.ones(2, requires_grad=True), trainable=True))
        with self.assertRaisesRegex(ValueError, "class count"):
            validate_label_state(label_state(torch.ones(1, 2, requires_grad=True), trainable=True), class_count=3)

    def test_objective_rejects_nonfinite_values(self):
        job = dlg()
        components = attack_components(job)
        components["loss"] = resolved_function(
            function=lambda reconstructed_gradients, **kwargs: reconstructed_gradients[0].sum() * float("nan"),
            name="nonfinite",
        )
        with self.assertRaisesRegex(FloatingPointError, "finite scalar"):
            run_attempt(attack_information(), components, OneStepObserver(), 0)

    def test_integer_initialization_is_rejected_before_optimization(self):
        components = attack_components(dlg())
        components["initialization_policy"] = resolved_function(
            function=lambda: torch.ones(1, 1, 4, 4, dtype=torch.long),
            name="integer_initialization",
        )
        with self.assertRaisesRegex(ValueError, "finite floating-point"):
            run_attempt(attack_information(), components, OneStepObserver(), 0)

    def test_positive_restart_count_is_required(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            optimization_attack({}, {}, OneStepObserver(), restarts=0)

    def test_effort_is_accumulated_across_restarts(self):
        job = dlg()
        job.optimizer_description = function_package("adam", {"lr": .01})
        observer = OneStepObserver()
        optimization_attack(attack_information(), attack_components(job), observer, restarts=2)
        self.assertEqual(observer.metrics["outer_iterations"], 2)
        self.assertEqual(observer.metrics["restarts_started"], 2)
        self.assertEqual(observer.metrics["objective_evaluations"], 4)
        self.assertEqual(observer.metrics["gradient_evaluations"], 4)
        self.assertGreaterEqual(observer.run_metrics["elapsed_seconds"], 0.)

    def test_all_nine_presets_execute_bounded_synthetic_smoke(self):
        presets = build_registries()["attacks"]
        self.assertEqual(len(presets), 9)
        for name, preset in presets.items():
            with self.subTest(attack=name):
                job = preset.build()
                if isinstance(job.label_policy, function_package):
                    job.label_policy.parameters.update(population_size=4, iterations=2)
                if job.optimizer_description.name == "lbfgs":
                    job.optimizer_description.parameters.update(max_iter=2, max_eval=3)
                observer = OneStepObserver()
                result = run_attempt(attack_information(), attack_components(job), observer, 0)
                self.assertEqual(result.shape, (1, 1, 4, 4))
                self.assertTrue(torch.isfinite(result).all())
                self.assertGreaterEqual(observer.metrics["objective_evaluations"], 2)
                if name not in {"dlg", "idlg"}:
                    self.assertGreaterEqual(result.min().item(), 0.)
                    self.assertLessEqual(result.max().item(), 1.)


if __name__ == "__main__":
    unittest.main()
