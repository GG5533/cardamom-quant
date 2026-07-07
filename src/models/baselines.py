"""Models. The rule of this repo: ML must beat the dumb seasonal rule or we
say so in public.

SeasonalBaseline is deliberately embarrassing to lose to — long in the lean
season (Mar–Jul, pre-harvest scarcity), mildly short otherwise. If a
regularized logistic regression can't beat it out-of-sample, that IS the
finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SeasonalBaseline:
    """Probability 0.5 +/- edge by season. No fitting on prices at all."""

    def __init__(self, edge: float = 0.10):
        self.edge = edge

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lean = X["lean_season"].to_numpy() > 0.5
        p_up = np.where(lean, 0.5 + self.edge, 0.5 - self.edge / 2)
        return np.column_stack([1 - p_up, p_up])


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000)),
        ]
    )


def make_gbm() -> HistGradientBoostingClassifier:
    # handles NaN natively; shallow + slow learning for small noisy data
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        l2_regularization=1.0, random_state=0,
    )


MODELS = {
    "seasonal_baseline": lambda: SeasonalBaseline(),
    "logistic": make_logistic,
    "gbm": make_gbm,
}
