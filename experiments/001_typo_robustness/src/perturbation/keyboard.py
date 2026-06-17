"""keyboard.py — QWERTY adjacency map (keyboard-neighbor selection policy)."""

from __future__ import annotations

_QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def _build_qwerty_neighbors() -> dict[str, str]:
    position: dict[str, tuple[int, int]] = {}

    for row_index, row in enumerate(_QWERTY_ROWS):
        for col_index, character in enumerate(row):
            position[character] = (row_index, col_index)

    neighbors: dict[str, set[str]] = {ch: set() for ch in position}

    for character, (row_index, col_index) in position.items():
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue

                neighbor_row = row_index + delta_row
                neighbor_col = col_index + delta_col

                if (0 <= neighbor_row < len(_QWERTY_ROWS)
                        and 0 <= neighbor_col < len(_QWERTY_ROWS[neighbor_row])):
                    neighbors[character].add(_QWERTY_ROWS[neighbor_row][neighbor_col])

    return {ch: "".join(sorted(adjacent)) for ch, adjacent in neighbors.items()}


QWERTY_NEIGHBORS: dict[str, str] = _build_qwerty_neighbors()
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def keyboard_neighbors(character: str) -> str:
    """QWERTY neighbors of character, case-preserved. Empty string if non-letter."""

    lower = character.lower()
    if lower not in QWERTY_NEIGHBORS:
        return ""

    neighbor_string = QWERTY_NEIGHBORS[lower]
    return neighbor_string.upper() if character.isupper() else neighbor_string
