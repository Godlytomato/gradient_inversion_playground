from validation.generator import generate_yamls

generate_yamls(
    attacks=None,
    defenses=None,
    models=["resnet18","lenet5"],
    datasets=["cifar10"],
    target=0.02,
)