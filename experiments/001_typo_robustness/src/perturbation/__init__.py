"""The perturbation sub-package: QWERTY keyboard adjacency and the four primitive
edits over Damerau-Levenshtein space, applied deterministically with a fully
reconstructible edit script.

Re-exports the full public surface of both modules so callers can import from
the sub-package rather than drilling into the individual files:

    from perturbation import perturb, Edit, PerturbationError
    from perturbation import keyboard_neighbors_of, QWERTY_NEIGHBORS
"""

from perturbation.keyboard import (
    QWERTY_PHYSICAL_ROWS,
    QWERTY_NEIGHBORS,
    keyboard_neighbors_of,
)

from perturbation.engine import (
    LOWERCASE_ALPHABET,
    PRIMITIVE_OPERATIONS,
    SELECTION_POLICIES,
    PERTURBATION_SCOPES,
    PerturbationError,
    Edit,
    damerau_levenshtein_distance,
    perturb,
    apply_edit_script,
)
