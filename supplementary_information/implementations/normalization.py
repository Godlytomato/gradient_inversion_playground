def normalization(
    context,
    data_def
):
    if data_def.mean is None or data_def.std is None:
        raise ValueError(
            f"Dataset '{data_def.name}' does not define normalization"
        )

    return {
        "mean": data_def.mean,
        "std": data_def.std,
        "input_space": data_def.input_space,
    }
