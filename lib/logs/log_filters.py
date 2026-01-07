# common_lib/logs/log_filters.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from typing import Iterable, Tuple


# def calc_date_range(*dfs: pd.DataFrame) -> Tuple[dt.date, dt.date]:
#     """複数の df から date の min/max をまとめて計算"""
#     dates = []
#     for df in dfs:
#         if "date" in df.columns:
#             dates.append(df["date"].dropna())
#     if not dates:
#         today = dt.date.today()
#         return today, today

#     all_dates = pd.concat(dates)
#     return all_dates.min(), all_dates.max()

def calc_date_range(*dfs: pd.DataFrame) -> Tuple[dt.date, dt.date]:
    """
    複数の df から date の min/max をまとめて計算。

    優先順位：
      1) "date" 列があればそれを使う
      2) なければ "ts"（ISO文字列想定）から日付を作って使う
      3) どちらも無ければ today,today
    """
    date_series_list = []

    for df in dfs:
        if df is None or getattr(df, "empty", True):
            continue

        # 1) 既に date 列があるなら最優先
        if "date" in df.columns:
            s = df["date"].dropna()
            if not s.empty:
                # date/datetime/str いずれでも受ける（失敗は NaT）
                sdt = pd.to_datetime(s, errors="coerce")
                sdate = sdt.dt.date.dropna()
                if not sdate.empty:
                    date_series_list.append(sdate)
            continue

        # 2) ts 列があるなら、そこから date を作る（JSONL の基本）
        if "ts" in df.columns:
            s = df["ts"].dropna()
            if not s.empty:
                # ts は "+09:00" を含む ISO を想定（JsonlLogger の ts）
                sdt = pd.to_datetime(s, errors="coerce", utc=True)

                # UTC→JST に寄せて date 化（JST基準で min/max したいので）
                try:
                    sdt = sdt.dt.tz_convert("Asia/Tokyo")
                except Exception:
                    # tz無し混在などで失敗したら、date だけ落とす
                    pass

                sdate = sdt.dt.date.dropna()
                if not sdate.empty:
                    date_series_list.append(sdate)
            continue

    if not date_series_list:
        today = dt.date.today()
        return today, today

    all_dates = pd.concat(date_series_list, ignore_index=True)
    # pandas Series の min/max は python の date を返す
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
