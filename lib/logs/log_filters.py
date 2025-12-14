# common_lib/logs/log_filters.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from typing import Iterable, Tuple


def calc_date_range(*dfs: pd.DataFrame) -> Tuple[dt.date, dt.date]:
    """複数の df から date の min/max をまとめて計算"""
    dates = []
    for df in dfs:
        if "date" in df.columns:
            dates.append(df["date"].dropna())
    if not dates:
        today = dt.date.today()
        return today, today

    all_dates = pd.concat(dates)
    return all_dates.min(), all_dates.max()


def apply_common_filters(
    df: pd.DataFrame,
    date_from: dt.date,
    date_to: dt.date,
    actions: Iterable[str] | None = None,
    pages: Iterable[str] | None = None,
) -> pd.DataFrame:
    """date / action / page で共通フィルタをかける"""
    if df.empty:
        return df.copy()

    mask = pd.Series(True, index=df.index)

    if "date" in df.columns:
        mask &= (df["date"] >= date_from) & (df["date"] <= date_to)

    if actions and "action" in df.columns:
        mask &= df["action"].isin(list(actions))

    if pages and "page" in df.columns:
        mask &= df["page"].isin(list(pages))

    return df[mask].copy()
