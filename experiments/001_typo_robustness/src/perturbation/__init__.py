"""perturbation — The typographical perturbation engine for Experiment 001.

Import from here rather than from sub-modules:

    from perturbation import perturb, Edit, apply_edit_script, PerturbationError
    from perturbation import Operation, SelectionPolicy, Scope, Unit, Regime
    from perturbation import damerau_levenshtein, single_edit_candidates
    from perturbation import keyboard_neighbors, QWERTY_NEIGHBORS, ALPHABET
"""

from .engine import (
    Edit,
    Operation,
    PerturbationError,
    Regime,
    Scope,
    SelectionPolicy,
    Unit,
    apply_edit_script,
    damerau_levenshtein,
    edited_words,
    perturb,
    single_edit_candidates,
)
from .keyboard import ALPHABET, QWERTY_NEIGHBORS, keyboard_neighbors

__all__ = [
    "Edit",
    "PerturbationError",
    "apply_edit_script",
    "edited_words",
    "Operation",
    "SelectionPolicy",
    "Scope",
    "Unit",
    "Regime",
    "perturb",
    "ALPHABET",
    "QWERTY_NEIGHBORS",
    "keyboard_neighbors",
    "damerau_levenshtein",
    "single_edit_candidates",
]
