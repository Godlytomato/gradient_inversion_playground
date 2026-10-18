from run_utils.runtime_context import runtime_context


def get_class_count(context: runtime_context) -> int:
    return context.dataset_definition.class_count