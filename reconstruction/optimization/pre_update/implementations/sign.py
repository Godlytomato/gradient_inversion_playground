def sign_update(
    reconstruction,
    state=None,
    context=None
):
    if reconstruction.grad is not None:
        reconstruction.grad.sign_()