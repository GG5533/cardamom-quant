"""SHAP interpretability — per-prediction attribution on top of the
permutation importance already in run.py.

Both attributions are reported side by side deliberately: permutation
importance answers "what does the model NEED", SHAP answers "what moved
THIS prediction". When the two disagree materially it usually means
correlated features — worth a line in the write-up either way.

shap is an optional dependency (pip install shap); everything degrades
gracefully without it.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def shap_values_for(model_name: str, fitted, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values for a fitted model from src/models/baselines.py.

    logistic  -> LinearExplainer on the scaled design matrix, mapped back
    gbm       -> TreeExplainer (native HistGradientBoosting support)
    Returns a frame of per-row SHAP values (same shape as X).
    """
    import shap  # optional heavy dep

    if model_name == "logistic":
        imputer = fitted.named_steps["impute"]
        scaler = fitted.named_steps["scale"]
        clf = fitted.named_steps["clf"]
        Xt = scaler.transform(imputer.transform(X))
        explainer = shap.LinearExplainer(clf, Xt)
        vals = explainer.shap_values(Xt)
    elif model_name == "gbm":
        explainer = shap.TreeExplainer(fitted)
        vals = explainer.shap_values(X.to_numpy())
        if isinstance(vals, list):  # older shap returns [class0, class1]
            vals = vals[1]
    else:
        raise ValueError(f"no SHAP path for model {model_name!r}")
    return pd.DataFrame(np.asarray(vals), index=X.index, columns=X.columns)


def mean_abs_shap(shap_df: pd.DataFrame) -> pd.Series:
    """Global importance: mean |SHAP| per feature, descending."""
    return shap_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")


def shap_report(model_name: str, fitted, X: pd.DataFrame, top: int = 15) -> str:
    try:
        sv = shap_values_for(model_name, fitted, X)
    except ImportError:
        return "shap not installed — `pip install shap` to enable attributions"
    ma = mean_abs_shap(sv).head(top)
    lines = [f"mean |SHAP| ({model_name}, {len(X)} rows):"]
    lines += [f"  {k:<22} {v:.4f}" for k, v in ma.items()]
    return "\n".join(lines)
