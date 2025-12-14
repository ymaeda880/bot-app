# common_lib/logs/log_io.py
from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import datetime as dt

import pandas as pd

JST = dt.timezone(dt.timedelta(hours=9), name="Asia/Tokyo")


def load_jsonl_logs(path: Path) -> pd.DataFrame:
    """JSONL ログを読み込んで ts/date/month/user を揃えた DataFrame を返す"""
    if not path.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ts → JST
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        df["ts"] = ts.dt.tz_convert("Asia/Tokyo")
        df["date"] = df["ts"].dt.date
        df["month"] = df["ts"].dt.strftime("%Y-%m")
    else:
        df["ts"] = pd.NaT
        df["date"] = pd.NaT
        df["month"] = None

    df["user"] = df.get("user", "(anonymous)").fillna("(anonymous)")
    return df


def load_sqlite_logs(db_path: Path, table: str = "logs") -> pd.DataFrame:
    """SQLite の logs テーブルを読み込んで ts/date/month/user を揃えた DataFrame を返す"""
    if not db_path.exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("Asia/Tokyo")
        df["ts"] = ts
        df["date"] = df["ts"].dt.date
        df["month"] = df["ts"].dt.strftime("%Y-%m")
    else:
        df["ts"] = pd.NaT
        df["date"] = pd.NaT
        df["month"] = None

    if "user" in df.columns:
        df["user"] = df["user"].fillna("(anonymous)")
    else:
        df["user"] = "(anonymous)"

    return df
