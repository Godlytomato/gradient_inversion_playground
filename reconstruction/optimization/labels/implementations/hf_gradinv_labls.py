import torch


from reconstruction.optimization.labels.label_policy_def import (

    label_state,

)


def hf_gradinv_labels(

    model,

    gradients,

    auxiliary_images,

    batch_size,

    population_size=100,

    iterations=100,

    feature_count=20,

):

    # The sign of an aggregate class gradient does not identify every class
    # present in a multi-image batch. Do not silently fix incomplete labels.
    if batch_size != 1:
        raise ValueError("HF-GradInv label inference supports batch_size=1 only")

    device = next(model.parameters()).device


    classifier = get_classifier_layer(model)


    fc_gradient = get_classifier_weight_gradient(

        model=model,

        gradients=gradients,

        classifier=classifier,

    )


    # ---------------------------------------------------------
    # Infer candidate labels.
    #
    # Prefer the classifier bias gradient when it exists. For
    # cross-entropy, especially with batch size 1, it provides a
    # cleaner class signal than averaging each weight-gradient row.
    #
    # If the classifier has no bias, fall back to the original
    # classifier-weight row-mean heuristic.
    # ---------------------------------------------------------

    bias_gradient = get_classifier_bias_gradient(
        model=model,
        gradients=gradients,
        classifier=classifier,
    )

    if bias_gradient is not None:

        candidate_score = bias_gradient

        candidate_classes = [
            class_index
            for class_index, value in enumerate(candidate_score)
            if value.item() < 0
        ]

    else:

        candidate_score = fc_gradient.mean(dim=1)

        candidate_classes = [
            class_index
            for class_index, value in enumerate(candidate_score)
            if value.item() < 0
        ]

    # If numerical effects or transformed gradients remove all
    # strict negatives, keep the lowest-scoring classes instead of
    # crashing before reconstruction begins.
    if not candidate_classes:

        count = min(
            batch_size,
            candidate_score.numel(),
        )

        candidate_classes = torch.topk(
            candidate_score,
            k=count,
            largest=False,
        ).indices.tolist()

    # There cannot be more distinct classes in the private batch
    # than there are samples. Keep the strongest candidates.
    if len(candidate_classes) > batch_size:

        candidate_classes = sorted(
            candidate_classes,
            key=lambda class_index:
                candidate_score[class_index].item()
        )

        candidate_classes = candidate_classes[
            :batch_size
        ]


    auxiliary_features = get_auxiliary_features(

        model=model,

        images=auxiliary_images,

        classifier=classifier,

    )


    reference = build_reference_statistics(

        features=auxiliary_features,

        feature_count=feature_count,

    )


    selected_gradient = fc_gradient[

        candidate_classes

    ]


    counts = estimate_label_counts(

        selected_gradient=selected_gradient,

        reference=reference,

        batch_size=batch_size,

        population_size=population_size,

        iterations=iterations,

    )


    if not validate_counts(

        counts=counts,

        batch_size=batch_size,

        class_count=len(candidate_classes),

    ):

        raise RuntimeError(

            "HF-GradInv produced invalid label counts"

        )


    labels = counts_to_labels(

        counts=counts,

        candidate_classes=candidate_classes,

        device=device,

    )


    return label_state(

        labels=labels,

        trainable=False,

    )


def get_classifier_layer(model):

    linear_layers = [

        module

        for module in model.modules()

        if isinstance(module, torch.nn.Linear)

    ]


    if not linear_layers:

        raise RuntimeError(

            "HF-GradInv requires a linear classifier layer"

        )


    return linear_layers[-1]


def get_classifier_weight_gradient(

    model,

    gradients,

    classifier,

):

    gradient_list = (

        gradients.grads

        if hasattr(gradients, "grads")

        else gradients

    )


    parameters = list(

        model.parameters()

    )


    for index, parameter in enumerate(parameters):

        if parameter is classifier.weight:

            return gradient_list[index].detach()


    raise RuntimeError(

        "Could not locate classifier weight gradient"

    )


def get_classifier_bias_gradient(
    model,
    gradients,
    classifier,
):
    """
    Return the final classifier bias gradient when available.
    Return None when the classifier has no bias.
    """

    if classifier.bias is None:
        return None

    gradient_list = (
        gradients.grads
        if hasattr(gradients, "grads")
        else gradients
    )

    parameters = list(
        model.parameters()
    )

    for index, parameter in enumerate(parameters):
        if parameter is classifier.bias:
            return gradient_list[index].detach()

    return None


def get_auxiliary_features(

    model,

    images,

    classifier,

):

    features = []


    def save_features(module, inputs, output):

        features.append(

            inputs[0].detach()

        )


    handle = classifier.register_forward_hook(

        save_features

    )


    device = next(model.parameters()).device

    images = images.to(device)


    try:

        with torch.no_grad():

            model(images)

    finally:

        handle.remove()


    if not features:

        raise RuntimeError(

            "Could not extract auxiliary features"

        )


    extracted = features[0]


    if extracted.ndim != 2:

        extracted = extracted.reshape(

            extracted.shape[0],

            -1,

        )


    return extracted


def build_reference_statistics(

    features,

    feature_count=20,

):

    if features.ndim != 2:

        features = features.reshape(

            features.shape[0],

            -1,

        )


    # HF-GradInv compares sorted feature magnitudes

    sorted_features = torch.sort(

        torch.abs(features),

        dim=1,

    )[0]


    average = sorted_features.mean(

        dim=0,

        keepdim=True,

    )


    variance = sorted_features.var(

        dim=0,

        unbiased=False,

    )


    feature_count = min(

        feature_count,

        sorted_features.shape[1],

    )


    feature_indices = torch.topk(

        variance,

        k=feature_count,

    ).indices


    return {

        "avg": average[:, feature_indices],

        "sort_var": feature_indices,

    }


def estimate_label_counts(

    selected_gradient,

    reference,

    batch_size,

    population_size,

    iterations,

):

    class_count = selected_gradient.shape[0]


    if class_count == 1:

        return torch.tensor(

            [batch_size],

            device=selected_gradient.device,

            dtype=torch.long,

        )


    reference_average = reference["avg"]

    feature_indices = reference["sort_var"]


    sorted_gradient = torch.sort(

        torch.abs(selected_gradient),

        dim=1,

    )[0]


    sorted_gradient = sorted_gradient[

        :,

        feature_indices

    ].reshape(1, -1)


    population = initialize_population(

        class_count=class_count,

        batch_size=batch_size,

        population_size=population_size,

        device=selected_gradient.device,

    )


    return run_evolution(

        population=population,

        iterations=iterations,

        reference_average=reference_average,

        sorted_gradient=sorted_gradient,

        batch_size=batch_size,

    )


def initialize_population(

    class_count,

    batch_size,

    population_size,

    device,

):

    if class_count > batch_size:

        raise ValueError(

            "Class count cannot exceed batch size"

        )


    population = torch.ones(

        (population_size, class_count),

        dtype=torch.long,

        device=device,

    )


    remaining = batch_size - class_count


    for i in range(population_size):

        if remaining == 0:

            continue


        assignments = torch.randint(

            0,

            class_count,

            (remaining,),

            device=device,

        )


        additions = torch.bincount(

            assignments,

            minlength=class_count,

        )


        population[i] += additions


    return population


def label_count_fitness(

    counts,

    reference_average,

    sorted_gradient,

    batch_size,

):

    expected = [

        count * reference_average / batch_size

        for count in counts

    ]


    expected = torch.cat(

        expected,

        dim=1,

    )


    similarity = (

        expected * sorted_gradient

    ).sum()


    denominator = (

        expected.norm(2)

        * sorted_gradient.norm(2)

    )


    epsilon = torch.finfo(

        expected.dtype

    ).eps


    denominator = denominator.clamp_min(

        epsilon

    )


    return similarity / denominator


def rescale_counts(

    counts,

    batch_size,

):

    counts = counts.clone().long()


    class_count = len(counts)


    if class_count == 1:

        counts[0] = batch_size

        return counts


    counts = torch.clamp(

        counts,

        min=1,

    )


    while counts.sum() > batch_size:

        candidates = torch.where(

            counts > 1

        )[0]


        if len(candidates) == 0:

            break


        index = candidates[

            torch.randint(

                len(candidates),

                (1,),

                device=counts.device,

            )

        ]


        counts[index] -= 1


    while counts.sum() < batch_size:

        index = torch.randint(

            0,

            class_count,

            (1,),

            device=counts.device,

        )


        counts[index] += 1


    return counts


def crossover_counts(

    parent_one,

    parent_two,

    batch_size,

):

    mask = torch.rand(

        parent_one.shape,

        device=parent_one.device,

    ) < 0.5


    child = torch.where(

        mask,

        parent_one,

        parent_two,

    )


    return rescale_counts(

        counts=child,

        batch_size=batch_size,

    )


def mutate_counts(

    counts,

):

    counts = counts.clone()


    if len(counts) <= 1:

        return counts


    donors = torch.where(

        counts > 1

    )[0]


    if len(donors) == 0:

        return counts


    donor = donors[

        torch.randint(

            len(donors),

            (1,),

            device=counts.device,

        )

    ].item()


    receivers = [

        index

        for index in range(len(counts))

        if index != donor

    ]


    receiver = receivers[

        torch.randint(

            len(receivers),

            (1,),

        ).item()

    ]


    counts[donor] -= 1

    counts[receiver] += 1


    return counts
def generate_child_population(
    parent_population,
    batch_size,
):
    population_size = parent_population.shape[0]
    permutation = torch.randperm(
        population_size,
        device=parent_population.device,
    )

    paired_population = parent_population[
        permutation
    ]
    child_population = parent_population.clone()

    for i in range(population_size):
        child = crossover_counts(
            parent_one=parent_population[i],
            parent_two=paired_population[i],
            batch_size=batch_size,
        )
        child_population[i] = mutate_counts(
            child
        )
    return child_population
def select_population(
    parent_population,
    child_population,
    reference_average,
    sorted_gradient,
    batch_size,
):
    population_size = parent_population.shape[0]
    fitness_scores = []
    for i in range(population_size):
        parent_fitness = label_count_fitness(
            counts=parent_population[i],
            reference_average=reference_average,
            sorted_gradient=sorted_gradient,
            batch_size=batch_size,
        )
        child_fitness = label_count_fitness(
            counts=child_population[i],
            reference_average=reference_average,
            sorted_gradient=sorted_gradient,
            batch_size=batch_size,
        )
        if child_fitness > parent_fitness:
            parent_population[i] = child_population[i]
            best_fitness = child_fitness
        else:
            best_fitness = parent_fitness
        fitness_scores.append(best_fitness.item())
    return parent_population, fitness_scores
def run_evolution(
    population,
    iterations,
    reference_average,
    sorted_gradient,
    batch_size,
):
    parent_population = population
    best_counts = parent_population[0].clone()
    best_fitness = float("-inf")
    for _ in range(iterations):
        child_population = generate_child_population(
            parent_population=parent_population,
            batch_size=batch_size,
        )
        parent_population, fitness_scores = select_population(
            parent_population=parent_population,
            child_population=child_population,
            reference_average=reference_average,
            sorted_gradient=sorted_gradient,
            batch_size=batch_size,
        )
        iteration_best = max(
            fitness_scores
        )
        best_index = fitness_scores.index(
            iteration_best
        )
        if iteration_best > best_fitness:
            best_fitness = iteration_best
            best_counts = parent_population[
                best_index
            ].clone()
    return best_counts
def counts_to_labels(
    counts,
    candidate_classes,
    device,
):
    labels = []
    for class_index, count in zip(
        candidate_classes,
        counts,
    ):
        labels.extend(
            [class_index] * int(count.item())
        )
    return torch.tensor(
        labels,
        device=device,
        dtype=torch.long,
    )
def validate_counts(
    counts,
    batch_size,
    class_count,
):
    if counts is None:
        return False
    if len(counts) != class_count:
        return False
    if any(
        int(count.item()) < 1
        for count in counts
    ):
        return False
    if sum(
        int(count.item())
        for count in counts
    ) != batch_size:
        return False
    return True
