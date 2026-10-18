
from transformations.application_selection_policy_defs import ranking_policy, draft_policy

# ranking policies
from transformations.ranking_implimentations.in_place import in_place
from transformations.ranking_implimentations.random import random
from transformations.ranking_implimentations.grad_mag import gradient_magnitude
from transformations.ranking_implimentations.jacobian_mag import jacobian_mag
from transformations.ranking_implimentations.max_svd import max_svd
from transformations.ranking_implimentations.min_svd import min_svd
from transformations.ranking_implimentations.exact_i2f import exact_i2f
from transformations.ranking_implimentations.soteria_sensitivity import soteria_sensitivity
from transformations.ranking_implimentations.graddefense_sensitivity import graddefense_sensitivity

# draft policies
from transformations.draft_implimentations.highest_draft import highest_draft
from transformations.draft_implimentations.lowest_draft import lowest_draft
from transformations.draft_implimentations.all import all_draft
from transformations.draft_implimentations.sensitivity_slices import sensitivity_slices
from transformations.draft_implimentations.highest_and_lowest import highest_lowest
from transformations.draft_implimentations.feature_columns import feature_columns


# ranking policy registry
ranking_policy_dict = {
    "graddefense_sensitivity": ranking_policy(
        name="graddefense_sensitivity",
        func=graddefense_sensitivity
    ),
    "soteria_sensitivity": ranking_policy(
        name="soteria_sensitivity",
        func=soteria_sensitivity
    ),
    "in_place": ranking_policy(
        name="in_place",
        func=in_place
    ),
    "random": ranking_policy(
        name="random",
        func=random
    ),
    "gradient_magnitude": ranking_policy(
        name="gradient_magnitude",
        func=gradient_magnitude
    ),
    "jacobian_magnitude": ranking_policy(
        name="jacobian_magnitude",
        func=jacobian_mag
    ),
    "max_svd": ranking_policy(
        name="max_svd",
        func=max_svd
    ),
    "min_svd": ranking_policy(
        name="min_svd",
        func=min_svd
    ),
    "exact_i2f": ranking_policy(
        name="exact_i2f",
        func=exact_i2f
    ),
}


# draft policy registry
draft_policy_dict = {
    "feature_columns": draft_policy(name="feature_columns", func=feature_columns),
    "all": draft_policy(
        name="all",
        func=all_draft
    ),
    "sensitivity_slices": draft_policy(
        name="sensitivity_slices",
        func=sensitivity_slices
    ),
    "highest_draft": draft_policy(
        name="highest_draft",
        func=highest_draft
    ),
    "lowest_draft": draft_policy(
        name="lowest_draft",
        func=lowest_draft
    ),
    "highest_and_lowest": draft_policy(
        name="highest_and_lowest",
        func=highest_lowest
    ),
}
