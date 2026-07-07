"""Purged + embargoed expanding-window walk-forward CV.

Why purging matters here: labels are FORWARD 5-day returns, so the last 5
training labels overlap the test window's information. Training on them is
look-ahead. Each fold therefore:

  * trains on an expanding window ending `purge` days BEFORE the test start
    (purge >= label horizon), and
  * additionally skips `embargo` days, guarding serial correlation bleed.

Folds are contiguous, non-overlapping test blocks marching forward in time —
the trading-desk deployment pattern, not random K-fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedWalkForward:
    n_splits: int = 6
    min_train: int = 500        # first fold's training size, samples
    purge: int = 5              # >= label horizon
    embargo: int = 5

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if n <= self.min_train + self.n_splits:  # degenerate
            raise ValueError(f"not enough samples ({n}) for walk-forward")
        test_size = (n - self.min_train) // self.n_splits
        if test_size < 20:
            raise ValueError(
                f"test blocks too small ({test_size}); reduce n_splits/min_train"
            )
        gap = self.purge + self.embargo
        for k in range(self.n_splits):
            test_start = self.min_train + k * test_size
            test_end = min(test_start + test_size, n)
            train_end = max(test_start - gap, 1)
            yield np.arange(0, train_end), np.arange(test_start, test_end)

    def n_test_samples(self, n: int) -> int:
        return sum(len(te) for _, te in self.split(n))
