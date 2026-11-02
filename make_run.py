from validation.generator import generate_yamls

generate_yamls(
    attacks=None,
    defenses=["none"],
    models=["lenet5"],
    datasets=["cifar10"],
    target=0.02,
)