from itertools import combinations
import math

import torch

from transformations.transform_def import grad_package


def ppga(
    package: grad_package,
    selection_masks: list[torch.Tensor],
    servers: int = 4,
    compromised_servers: int = 3,
    selection_number: int = 0,
    block_size: int = 32,
    seed: int = 2026,
    max_condition: float = 100.0,
    max_tries: int = 100,
    batch_blocks: int = 1024,
) -> grad_package:
    """
    Encode gradients into b coefficients per block, discard coefficients
    owned by uncompromised servers, and decode the visible contribution.

    selection_number is the zero-based index into:
        itertools.combinations(range(servers), compromised_servers)

    The same seed and batch_blocks produce the same encoding across coalition
    runs. The only change between them is which servers are compromised.

    All linear algebra runs on the CPU in float64. Blocks are processed in
    batches to avoid one Python loop / decomposition per gradient block.
    """
    if servers < 2:
        raise ValueError("servers must be at least 2")
    if not 0 <= compromised_servers <= servers:
        raise ValueError("compromised_servers must be between 0 and servers")
    if block_size < servers:
        raise ValueError("block_size must be at least servers")
    if max_condition < 1 or not math.isfinite(max_condition):
        raise ValueError("max_condition must be finite and at least 1")
    if max_tries < 1:
        raise ValueError("max_tries must be positive")
    if batch_blocks < 1:
        raise ValueError("batch_blocks must be positive")
    if not package.grads:
        raise ValueError("grad_package must contain gradients")
    if len(selection_masks) != len(package.grads):
        raise ValueError("selection_masks must match gradients")
    if any(
        mask.shape != grad.shape or not torch.all(mask != 0).item()
        for mask, grad in zip(selection_masks, package.grads)
    ):
        raise ValueError("PPGA requires the 'all' draft (every gradient selected)")

    coalition_count = math.comb(servers, compromised_servers)
    if not 0 <= selection_number < coalition_count:
        raise ValueError(
            f"selection_number must be in [0, {coalition_count - 1}]"
        )

    compromised = tuple(
        combinations(range(servers), compromised_servers)
    )[selection_number]

    rng = torch.Generator(device="cpu").manual_seed(seed)

    original_grads = package.grads
    sizes = [grad.numel() for grad in original_grads]
    original = torch.cat([
        grad.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        for grad in original_grads
    ])

    b = block_size
    total_columns = b + servers
    padding = (-original.numel()) % b
    if padding:
        padded = torch.nn.functional.pad(original, (0, padding))
    else:
        padded = original

    blocks = padded.view(-1, b)
    decoded_blocks = torch.empty_like(blocks)

    # Each decoder column belongs to exactly one parameter server.
    owners = torch.arange(total_columns) % servers
    server_columns = [
        torch.nonzero(owners == server).flatten()
        for server in range(servers)
    ]
    compromised_mask = torch.zeros(servers, dtype=torch.bool)
    compromised_mask[list(compromised)] = True

    for start in range(0, blocks.shape[0], batch_blocks):
        stop = min(start + batch_blocks, blocks.shape[0])
        pending = torch.arange(start, stop)

        # Retry only the blocks whose randomly selected square decoder
        # failed the conditioning threshold. Others are already finished.
        for _ in range(max_tries):
            count = pending.numel()
            if count == 0:
                break

            decoders = torch.randn(
                (count, b, total_columns),
                generator=rng,
                dtype=torch.float64,
            ) / math.sqrt(b)

            # Choose >= 1 column from each server for every block.
            required = torch.stack([
                columns[torch.randint(
                    columns.numel(), (count,), generator=rng
                )]
                for columns in server_columns
            ], dim=1)

            # Fill the remaining b-servers slots without replacement.
            scores = torch.rand((count, total_columns), generator=rng)
            scores.scatter_(1, required, float("inf"))
            extras = scores.topk(
                k=b - servers, dim=1, largest=False
            ).indices
            selected = torch.cat((required, extras), dim=1).sort(dim=1).values

            selected_decoders = decoders.gather(
                2, selected[:, None, :].expand(-1, b, -1)
            )

            # A continuous random matrix is full spark with probability 1
            # in exact arithmetic. Only the SELECTED b-column submatrix is
            # numerically checked; we do not certify all combinations.
            condition = torch.linalg.cond(selected_decoders)
            accepted = torch.isfinite(condition) & (condition <= max_condition)

            if accepted.any().item():
                accepted_indices = pending[accepted]
                accepted_selected = selected[accepted]
                accepted_decoders = selected_decoders[accepted]

                coefficients = torch.linalg.solve(
                    accepted_decoders,
                    blocks[accepted_indices].unsqueeze(-1),
                ).squeeze(-1)

                # Zero the coefficients belonging to honest servers,
                # then decode the visible coefficients into gradient space.
                visible = compromised_mask[owners[accepted_selected]]
                visible_coefficients = coefficients.masked_fill(~visible, 0.0)
                partial = torch.bmm(
                    accepted_decoders, visible_coefficients.unsqueeze(-1)
                ).squeeze(-1)
                decoded_blocks[accepted_indices] = partial

            pending = pending[~accepted]

        if pending.numel():
            raise RuntimeError(
                f"Could not find well-conditioned PPGA encodings for "
                f"{pending.numel()} blocks after {max_tries} attempts; "
                "increase max_condition or max_tries"
            )

    decoded = decoded_blocks.reshape(-1)[:original.numel()]

    output_grads = []
    offset = 0
    for grad, size in zip(original_grads, sizes):
        output_grads.append(
            decoded[offset:offset + size]
            .reshape(grad.shape)
            .to(dtype=grad.dtype, device=grad.device)
        )
        offset += size

    # Partial decoding is not coordinate-wise pruning, so gradients have
    # no missing-coordinate mask from the attack's perspective.
    return grad_package(grads=output_grads, missing_mask=None)