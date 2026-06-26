"""Uniform progress reporting for long-running pipeline steps.

Uses tqdm when available — it auto-detects a Jupyter / Colab environment and
renders a rich notebook widget, or falls back to a terminal progress bar on the
cluster. When tqdm is absent, a plain-text reporter prints one status line every
``print_interval_seconds`` seconds so the user always knows something is
happening.

All reporters are usable as context managers:

    with ProgressBar(total=len(items), description="fetching items") as progress:
        for item in items:
            process(item)
            progress.advance()

Or stand-alone (remember to call ``close()`` when done):

    progress = ProgressBar(total=total_count, description="generating")
    for batch in batches:
        run_batch(batch)
        progress.advance(count=len(batch))
    progress.close()
"""

from __future__ import annotations

import time
from typing import Optional


_PRINT_INTERVAL_SECONDS: float = 5.0


class ProgressBar:
    """A tqdm-backed progress bar that degrades gracefully when tqdm is absent.

    Parameters
    ----------
    total:
        Total number of work units to complete.
    description:
        Short label shown to the left of the bar (e.g. ``"generating"``).
    print_interval_seconds:
        When tqdm is absent, print a status line at most this often.
        Defaults to ``_PRINT_INTERVAL_SECONDS``.
    """

    def __init__(
            self,
            total: int,
            description: str = "",
            print_interval_seconds: float = _PRINT_INTERVAL_SECONDS,
    ) -> None:

        self._total = total
        self._description = description
        self._print_interval_seconds = print_interval_seconds
        self._completed = 0
        self._start_monotonic = time.monotonic()
        self._last_print_monotonic = self._start_monotonic
        self._tqdm_bar: Optional[object] = None

        try:
            import tqdm as _tqdm
            self._tqdm_bar = _tqdm.tqdm(
                total=total,
                desc=description,
                unit="item",
                dynamic_ncols=True,
            )
        except ImportError:
            self._print_status_line()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advance(self, count: int = 1) -> None:
        """Record ``count`` completed work units and refresh the display."""

        self._completed += count

        if self._tqdm_bar is not None:
            self._tqdm_bar.update(count)  # type: ignore[attr-defined]
            return

        now = time.monotonic()
        if now - self._last_print_monotonic >= self._print_interval_seconds:
            self._print_status_line()
            self._last_print_monotonic = now

    def set_description(self, description: str) -> None:
        """Update the label shown to the left of the bar."""

        self._description = description
        if self._tqdm_bar is not None:
            self._tqdm_bar.set_description(description)  # type: ignore[attr-defined]

    def close(self) -> None:
        """Finalise the bar and print a completion line (plain-text path)."""

        if self._tqdm_bar is not None:
            self._tqdm_bar.close()  # type: ignore[attr-defined]
            return

        elapsed_seconds = time.monotonic() - self._start_monotonic
        print(
            f"  {self._description}  "
            f"done — {self._completed} / {self._total}  "
            f"[{elapsed_seconds:.0f}s]",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_status_line(self) -> None:
        elapsed_seconds = time.monotonic() - self._start_monotonic
        percent_complete = 100 * self._completed / max(1, self._total)
        print(
            f"  {self._description}  "
            f"{self._completed} / {self._total}  "
            f"({percent_complete:.0f}%)  "
            f"[{elapsed_seconds:.0f}s]",
            flush=True,
        )
