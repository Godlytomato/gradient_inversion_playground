"""Generate one DLG/LeNet reconstruction with a 0.90 CLIP target."""
import argparse
from pathlib import Path

from run_def import best_selection_description, function_package
from validation.generator import generate_run, save_yaml
from validation.registries.registry_builder import build_registries
from validation.validator import validate_run


def demo_description(*, iterations=1000, threshold=0.90, temperature=1.0, model_name="ViT-B/32"):
    presets = build_registries()
    description = generate_run(
        presets["attacks"]["dlg"], presets["defenses"]["none"],
        presets["models"]["lenet5"], presets["datasets"]["cifar10"], target=0.02,
    )
    description.name = "clip_confidence_demo"
    # DLG still jointly optimizes pixels and unknown labels. Smooth random
    # pixels avoid retaining high-frequency initialization noise in LeNet's
    # weakly observed directions; no private image is used for initialization.
    description.attack.initialization_policy = "smooth_uniform"
    description.attack.optimizer_description.parameters.update(
        tolerance_grad=1e-12, tolerance_change=1e-15,
    )
    # A positive constant preserves the squared gradient-matching objective's
    # minimizers while keeping its tiny gradients useful to LBFGS.
    description.attack.loss_mods = [function_package("scale", {"scale": 1e6})]
    config = description.observer
    config.metrics.append(function_package("clip_confidence", {
        "model_name": model_name, "temperature": temperature,
    }))
    config.success_policies = [function_package("target_clip_confidence", {"threshold": threshold})]
    config.fail_policies = [function_package("max_iterations", {"limit": iterations})]
    config.logging.metrics.append("clip_confidence")
    # The saved reconstruction must be the image that satisfied the CLIP rule.
    config.best_selection = best_selection_description("clip_confidence", "maximize")
    config.evaluation_target_mse = None
    config.evaluation_target_clip_confidence = threshold
    validate_run(description)
    return description


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo_schedule/clip_confidence"))
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Safety cap; stop earlier when correct-label CLIP confidence reaches the target")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip-model", default="ViT-B/32", help="OpenAI CLIP model name or local checkpoint path")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    description = demo_description(iterations=args.iterations, threshold=args.threshold,
                                   temperature=args.temperature, model_name=args.clip_model)
    # Never silently run unrelated YAML files alongside this single demo.
    expected = args.output / f"{description.name}.yaml"
    if args.output.exists():
        siblings = {path for path in args.output.iterdir() if path.suffix in {".yaml", ".yml"}}
        if siblings - {expected}:
            parser.error("Demo output directory contains other experiments; choose a separate --output")
    path = save_yaml(description, args.output)
    print(f"Generated demo: {path}", flush=True)
    if args.run:
        from run import main as run_main
        return run_main(["--schedule", str(args.output), "--strict-errors"])
    print(f"Run with: python run.py --schedule {args.output} --strict-errors")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
