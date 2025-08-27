"""Stub for the competition's custom Trendyol group AUC metric.

The original `trendyol_custom_auc.py` file distributed by the organizers is NOT
included in this repository due to data / licensing restrictions.

Expected public interface:
    score(solution_df, submission_df, session_col) -> float

Parameters
----------
solution_df : pd.DataFrame
    Must contain at minimum columns:
        session_id, ordered_items, clicked_items, all_items
    Each of those item columns is a single space‑separated string of content ids.
submission_df : pd.DataFrame
    Must contain columns:
        session_id, prediction
    Where `prediction` is a space‑separated string (ranked list) of content ids.
session_col : str
    Name of the session id column (typically 'session_id').

Usage
-----
Place the official `trendyol_custom_auc.py` (provided by the competition) next
to this stub OR update imports in notebooks to point to your local path. This
stub deliberately raises NotImplementedError to avoid silently using an
incorrect surrogate metric.

Optional Approximation
----------------------
For experimentation, an *approximate* session‑level ranking quality measure
(`approx_group_map`) is provided. It is NOT the official metric; reference only
as a placeholder when the official script is unavailable.
"""
from __future__ import annotations

from typing import Iterable
import pandas as pd


def score(solution_df: pd.DataFrame, submission_df: pd.DataFrame, session_col: str) -> float:  # pragma: no cover - intentionally unimplemented
    """Official competition metric placeholder.

    Replace this stub with the authentic implementation provided by the
    organizers. Do NOT attempt to rely on the fallback approximation for final
    evaluation.
    """
    raise NotImplementedError(
        "Official competition metric not included. Add original 'trendyol_custom_auc.py' locally (untracked) and import from there."
    )


# ----------------------------- OPTIONAL APPROX ---------------------------------
def _split_items(col: pd.Series) -> list[list[str]]:
    return [x.split() if isinstance(x, str) and x else [] for x in col.tolist()]


def approx_group_map(solution_df: pd.DataFrame, submission_df: pd.DataFrame, session_col: str) -> float:
    """Approximate mean average precision over sessions.

    This is NOT the official metric. Positives are union of ordered + clicked
    items (ordered given implicit higher weight via duplicate inclusion).
    """
    sol = solution_df.merge(submission_df, on=session_col, how="inner")
    if sol.empty:
        return 0.0

    ordered_lists = _split_items(sol.get("ordered_items", pd.Series([])))
    clicked_lists = _split_items(sol.get("clicked_items", pd.Series([])))
    preds = _split_items(sol.get("prediction", pd.Series([])))

    total = 0.0
    n_sessions = 0
    for ord_items, clk_items, pred in zip(ordered_lists, clicked_lists, preds):
        if not pred:
            continue
        # Build relevance dict; ordered weight 2, clicked weight 1 if not ordered
        rel = {}
        for c in clk_items:
            rel[c] = max(rel.get(c, 0), 1)
        for c in ord_items:
            rel[c] = max(rel.get(c, 0), 2)
        if not rel:
            continue
        hits = 0.0
        ap = 0.0
        for rank, cid in enumerate(pred, start=1):
            if cid in rel:
                hits += rel[cid]
                ap += hits / rank
        denom = sum(rel.values())
        if denom > 0:
            total += ap / denom
            n_sessions += 1
    return total / n_sessions if n_sessions else 0.0


__all__ = ["score", "approx_group_map"]
