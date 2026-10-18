# transformations/draft_implimentations/sensitivity_slices.py

import math
import random

import torch


# Adapted from:
#
# Junxiao Wang, Song Guo, Xin Xie, Heng Qi.
# "Protect Privacy from Gradient Leakage Attack in Federated Learning."
# IEEE INFOCOM 2022.
#
# Official implementation:
# https://github.com/wangjunxiao/GradDefense
#
# Relevant original implementation:
# GradDefense/perturb.py
#
# The published implementation:
#   1. divides gradient tensors into equal-sized "atoms",
#   2. weights atoms using layer sensitivity,
#   3. groups atoms into approximately equal-sensitivity slices,
#   4. randomly selects slices for perturbation.
#
# This adaptation stops after step 4 and returns selection masks.
# Noise construction/application is handled by this project's
# transformation pipeline.


def sensitivity_slices(
    scores: list[torch.Tensor],
    slices_num: int,
    perturb_slices_num: int,
    seed: int | None = None,
    **kwargs
) -> list[torch.Tensor]:

    if len(scores) == 0:
        return []

    if slices_num <= 0:
        raise ValueError(
            "slices_num must be greater than zero"
        )

    if perturb_slices_num < 0:
        raise ValueError(
            "perturb_slices_num cannot be negative"
        )

    if perturb_slices_num > slices_num:
        raise ValueError(
            "perturb_slices_num cannot exceed slices_num"
        )

    # ---------------------------------------------------------
    # Recover one sensitivity value for each parameter tensor.
    #
    # graddefense_sensitivity produces tensors filled with one
    # layer sensitivity value, so each tensor should be uniform.
    # ---------------------------------------------------------

    layer_sensitivity = []
    layer_sizes = []

    for score in scores:

        if score.numel() == 0:
            raise ValueError(
                "Sensitivity score tensors cannot be empty"
            )

        flat = score.reshape(-1)

        sensitivity = flat[0]

        # This draft policy expects GradDefense-style
        # layerwise sensitivity rather than coordinate scores.
        if not torch.allclose(
            flat,
            torch.full_like(flat, sensitivity)
        ):
            raise ValueError(
                "sensitivity_slices requires one "
                "sensitivity value per gradient tensor"
            )

        layer_sensitivity.append(
            float(sensitivity.detach().cpu())
        )

        if not math.isfinite(layer_sensitivity[-1]) or layer_sensitivity[-1] < 0:
            raise ValueError("Layer sensitivity must be finite and nonnegative")

        layer_sizes.append(
            score.numel()
        )

    # ---------------------------------------------------------
    # Published GradDefense uses the greatest common divisor
    # of layer parameter counts as the atom size.
    # ---------------------------------------------------------

    atom_size = layer_sizes[0]

    for size in layer_sizes[1:]:
        atom_size = math.gcd(
            atom_size,
            size
        )

    if atom_size <= 0:
        raise RuntimeError(
            "Could not determine GradDefense atom size"
        )

    # ---------------------------------------------------------
    # Build ordered atoms.
    #
    # Each layer's total sensitivity is divided equally among
    # all atoms belonging to that layer, matching the original
    # GradDefense implementation.
    # ---------------------------------------------------------

    atoms = []

    for layer_index, (
        size,
        sensitivity
    ) in enumerate(
        zip(
            layer_sizes,
            layer_sensitivity
        )
    ):

        atom_count = size // atom_size

        atom_sensitivity = (
            sensitivity / atom_count
        )

        for atom_index in range(atom_count):

            start = atom_index * atom_size
            end = start + atom_size

            atoms.append(
                (
                    layer_index,
                    start,
                    end,
                    atom_sensitivity
                )
            )

    if slices_num > len(atoms):
        raise ValueError(
            "slices_num cannot exceed the number "
            "of available gradient atoms"
        )

    # ---------------------------------------------------------
    # Group sequential atoms into approximately equal-
    # sensitivity slices.
    #
    # The original GradDefense implementation uses a simple
    # sequential bin-packing procedure based on:
    #
    # sum(layer sensitivities) / slices_num
    # ---------------------------------------------------------

    target_sensitivity = (
        sum(layer_sensitivity) / slices_num
    )

    slices = [
        []
        for _ in range(slices_num)
    ]

    current_slice = 0
    current_sensitivity = 0.0

    for atom_index, atom in enumerate(atoms):

        atom_sensitivity = atom[3]

        # Leave all remaining atoms for the final slice.
        if current_slice == slices_num - 1:

            slices[current_slice].append(atom)
            continue

        # Move to the next slice when adding another atom
        # would cross the target and this slice already
        # contains at least one atom.
        remaining_atoms = len(atoms) - atom_index
        remaining_slices = slices_num - current_slice - 1
        if slices[current_slice] and (
            current_sensitivity + atom_sensitivity >= target_sensitivity
            # Reserve at least one atom for every remaining slice, even if
            # a dominant early atom exhausted most of the sensitivity.
            or remaining_atoms <= remaining_slices
        ):

            current_slice += 1
            current_sensitivity = 0.0

        slices[current_slice].append(atom)

        current_sensitivity += atom_sensitivity

    if not all(slices):
        raise RuntimeError("GradDefense partition produced an empty slice")

    # ---------------------------------------------------------
    # Randomly select slices for perturbation.
    #
    # The published code uses reservoir sampling. Python's
    # random.sample gives the same required result here:
    # uniformly select perturb_slices_num distinct slices.
    # ---------------------------------------------------------

    rng = random.Random(seed)

    selected_slices = set(
        rng.sample(
            range(slices_num),
            perturb_slices_num
        )
    )

    # ---------------------------------------------------------
    # Convert selected slices back into masks shaped exactly
    # like the original gradient/score tensors.
    # ---------------------------------------------------------

    masks = [
        torch.zeros_like(
            score,
            dtype=torch.bool
        )
        for score in scores
    ]

    flat_masks = [
        mask.reshape(-1)
        for mask in masks
    ]

    for slice_index in selected_slices:

        for (
            layer_index,
            start,
            end,
            _
        ) in slices[slice_index]:

            flat_masks[layer_index][
                start:end
            ] = True

    return masks
