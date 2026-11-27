from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from models.lenet import LeNet5, change_channels


# ---------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------
SEED = 2026
LEARNING_RATE = 0.05
BATCH_SIZE = 32
STEPS = 30
PROBE_INDEX = 0       # fixed held-out test image
DATA_ROOT = Path("data")
OUTPUT_DIR = Path("results") / "gradient_change"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def flatten(tensors):
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def experiment_step(model, train_x, train_y, probe_x, probe_y, lr):
    """Measure prediction accuracy for one exact, momentum-free SGD step."""
    params = list(model.parameters())

    # Actual training update, evaluated at the CURRENT parameters.
    train_loss = F.cross_entropy(model(train_x), train_y)
    train_grads = torch.autograd.grad(train_loss, params)
    delta = [(-lr * gradient).detach() for gradient in train_grads]

    # Gradient from the SAME fixed probe image before the update.
    probe_loss = F.cross_entropy(model(probe_x), probe_y)
    g_before = torch.autograd.grad(probe_loss, params, create_graph=True)

    # H_theta * delta, without forming the Hessian matrix.
    hvp = torch.autograd.grad(
        outputs=g_before,
        inputs=params,
        grad_outputs=delta,
    )
    g_predicted = flatten([g + h for g, h in zip(g_before, hvp)])
    g_old = flatten(g_before)

    # Apply EXACTLY the SGD update used in the prediction.
    with torch.no_grad():
        for parameter, change in zip(params, delta):
            parameter.add_(change)

    # Ground truth: recompute the probe gradient at the UPDATED model.
    new_probe_loss = F.cross_entropy(model(probe_x), probe_y)
    g_actual = flatten(torch.autograd.grad(new_probe_loss, params))

    denom = g_actual.norm().clamp_min(1e-12)
    baseline_relative = (g_old - g_actual).norm() / denom
    predicted_relative = (g_predicted - g_actual).norm() / denom
    prediction_mse = (g_predicted - g_actual).square().mean()

    return {
        "training_loss": train_loss.item(),
        "parameter_update_norm": flatten(delta).norm().item(),
        "baseline_relative_error": baseline_relative.item(),
        "predicted_relative_error": predicted_relative.item(),
        "predicted_gradient_mse": prediction_mse.item(),
        "improvement_factor": (baseline_relative / predicted_relative.clamp_min(1e-12)).item(),
    }


def main():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
    test_data = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
    loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )

    probe_x, probe_y = test_data[PROBE_INDEX]
    probe_x = probe_x.unsqueeze(0).to(DEVICE)
    probe_y = torch.tensor([probe_y], dtype=torch.long, device=DEVICE)

    model = LeNet5()
    change_channels(model, 3)  # original LeNet in the repo accepts 1-channel images
    model = model.to(DEVICE)
    model.train()

    # Initialize LazyLinear before collecting the model parameters.
    with torch.no_grad():
        model(probe_x)

    rows = []
    for step, (images, labels) in enumerate(loader, start=1):
        if step > STEPS:
            break
        stats = experiment_step(
            model,
            images.to(DEVICE),
            labels.to(DEVICE),
            probe_x,
            probe_y,
            LEARNING_RATE,
        )
        stats = {"step": step, **stats}
        rows.append(stats)
        print(
            f"step={step:02d} loss={stats['training_loss']:.4f} "
            f"no-change={stats['baseline_relative_error']:.6g} "
            f"HVP={stats['predicted_relative_error']:.6g} "
            f"improvement={stats['improvement_factor']:.2f}x"
        )

    csv_path = OUTPUT_DIR / "measurements.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        [row["step"] for row in rows],
        [row["baseline_relative_error"] for row in rows],
        label="Assume gradient unchanged",
    )
    plt.semilogy(
        [row["step"] for row in rows],
        [row["predicted_relative_error"] for row in rows],
        label="First-order Hessian prediction",
    )
    plt.xlabel("Training step")
    plt.ylabel("Relative L2 error vs actual gradient")
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "prediction_error.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"\nSaved {csv_path} and {plot_path}")


if __name__ == "__main__":
    main()