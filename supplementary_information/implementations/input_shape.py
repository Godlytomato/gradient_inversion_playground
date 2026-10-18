from run_utils.runtime_context import runtime_context


def get_input_shape(context: runtime_context) -> tuple[int, ...]:
    return (
        context.dataset_definition.input_channels,
        *context.dataset_definition.input_dimensions
    )