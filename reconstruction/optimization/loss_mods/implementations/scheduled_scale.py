def scheduled_scale(
    attack_loss,
    state=None,
    initial_scale=1.0,
    final_scale=1.0,
    transition_iteration=0
):

    if state is None:
        return attack_loss * initial_scale

    if state.iteration < transition_iteration:
        scale = initial_scale
    else:
        scale = final_scale

    return attack_loss * scale