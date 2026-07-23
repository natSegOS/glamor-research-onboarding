"""The QWERTY keyboard adjacency graph used by the keyboard-neighbor typo policy.

Provenance
----------
The keyboard-neighbor substitution policy follows MulTypo (Zhao et al., 2025,
arXiv:2510.09536), whose "replacement" operation replaces a single character
with a neighboring key on the language-specific keyboard layout. We implement
the English QWERTY layout. See docs/PROVENANCE.md §2.2 for the exact fidelity
statement (which operations follow the paper and which deliberately differ).

This module has no internal dependencies; it is the bottom of the import graph.
"""

from __future__ import annotations


# The three physical letter rows of a US-English QWERTY keyboard. We model only
# letters; digits and punctuation are handled by their own policies (numeric
# tokens are protected by default, and whitespace/punctuation have dedicated
# policies in the perturbation engine).
QWERTY_PHYSICAL_ROWS: tuple[str, ...] = (
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
)


def _build_adjacency_graph() -> dict[str, str]:
    """Return a mapping from each lowercase letter to the sorted string of its
    physically adjacent keys (the eight-neighbor Moore neighborhood on the
    staggered row grid, clipped to keys that actually exist).

    The result is computed once at import time and exposed as
    ``QWERTY_NEIGHBORS``. We sort the neighbor strings so the structure is
    deterministic and easy to assert against in tests.
    """
    position_of_key: dict[str, tuple[int, int]] = {}
    for row_index, row in enumerate(QWERTY_PHYSICAL_ROWS):
        for column_index, key in enumerate(row):
            position_of_key[key] = (row_index, column_index)

    neighbors_of_key: dict[str, set[str]] = {key: set() for key in position_of_key}

    for key, (row_index, column_index) in position_of_key.items():
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):

                is_the_key_itself = (row_offset == 0 and column_offset == 0)
                if is_the_key_itself:
                    continue

                neighbor_row = row_index + row_offset
                neighbor_column = column_index + column_offset

                row_exists = 0 <= neighbor_row < len(QWERTY_PHYSICAL_ROWS)
                if not row_exists:
                    continue

                column_exists = 0 <= neighbor_column < len(QWERTY_PHYSICAL_ROWS[neighbor_row])
                if not column_exists:
                    continue

                neighbors_of_key[key].add(QWERTY_PHYSICAL_ROWS[neighbor_row][neighbor_column])

    return {key: "".join(sorted(neighbor_set))
            for key, neighbor_set in neighbors_of_key.items()}


# Public mapping: lowercase letter -> sorted string of adjacent lowercase keys.
QWERTY_NEIGHBORS: dict[str, str] = _build_adjacency_graph()


def keyboard_neighbors_of(character: str) -> str:
    """Return the QWERTY-adjacent keys of ``character``, preserving its case.

    Returns the empty string for any non-letter (digits, punctuation, space),
    which the caller treats as "no keyboard-neighbor substitution available
    here" and skips.
    """
    lowercase_character = character.lower()

    if lowercase_character not in QWERTY_NEIGHBORS:
        return ""

    neighbors = QWERTY_NEIGHBORS[lowercase_character]
    return neighbors.upper() if character.isupper() else neighbors
