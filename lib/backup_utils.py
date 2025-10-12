# lib/backup_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Optional, List
from datetime import datetime, timezone
import shutil
import json
import numpy as np

# ===== 基本ユーティリティ =====
def timestamp() -> str:
    # 例: 20251011-093012
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")

def backup_dir_for(backup_root: Path, backend: str, shard_id: str,
                   ts: Optional[str] = None, *, label: Optional[str] = None) -> Path:
    """
    バックアップ保存先フォルダ
      例: backup_root/backend/shard_id/20251010-120000-Home[-1-Rollback|-2-PostOp]
    """
    if ts is None:
        ts = timestamp()

    # secrets.toml から location を取得（無くても落とさない）
    loc = "unknown"
    try:
        import streamlit as st
        loc = st.secrets["env"]["location"]
    except Exception:
        pass

    # ★ 並び順キー: 1 < 2 なので、reverse=True の並びで 2(=PostOp) が先頭 = 最新扱い
    label_alias = None
    if label:
        order_map = {
            "Rollback": "1-Rollback",
            "PostOp":   "2-PostOp",
        }
        label_alias = order_map.get(label, label)

    suffix = f"-{loc}"
    if label_alias:
        suffix += f"-{label_alias}"

    return backup_root / backend / shard_id / f"{ts}{suffix}"


# ====== バックアップ/復元 本体 ======
def _copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False

def backup_all_local(shard_dir: Path, backup_root: Path, backend: str, shard_id: str, *, label: Optional[str] = None) -> Tuple[List[str], Path]:
    """
    シャード内の主要ファイルをまとめて保存:
      - meta.jsonl / vectors.npy / processed_files.json / （その他 .jsonl/.npy も拾う）
    label を指定するとディレクトリ名が …-<loc>-<label> になる（例: -Rollback / -PostOp）
    """
    bdir = backup_dir_for(backup_root, backend, shard_id, label=label)
    bdir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []

    # 既知の主要ファイル
    for name in ("meta.jsonl", "vectors.npy", "processed_files.json"):
        if _copy_if_exists(shard_dir / name, bdir / name):
            copied.append(name)

    # 念のため同階層の .jsonl/.npy も保全（重複は上のcopy2で上書き）
    for p in shard_dir.glob("*.jsonl"):
        if p.name not in copied and _copy_if_exists(p, bdir / p.name):
            copied.append(p.name)
    for p in shard_dir.glob("*.npy"):
        if p.name not in copied and _copy_if_exists(p, bdir / p.name):
            copied.append(p.name)

    return copied, bdir


def list_backup_dirs_local(backup_root: Path, backend: str, shard_id: str, *, include_rollback: bool = True) -> List[Path]:
    """
    シャードのバックアップディレクトリ一覧を新しい順に返す。
    include_rollback=False のとき、名前末尾に '-Rollback' を含むフォルダを除外。
    """
    base = backup_root / backend / shard_id
    if not base.exists():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not include_rollback:
        dirs = [p for p in dirs if "-Rollback" not in p.name]
    # フォルダ名の辞書順≒時間降順（フォーマットが %Y%m%d-%H%M%S-<loc>… なので）
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def preview_backup_local(bdir: Path):
    """
    バックアップフォルダ内の主要ファイルを簡易プレビュー用の DataFrame にする。
    呼び出し側（Streamlit）で pd.DataFrame に渡す前提のレコード（辞書）を返す。
    """
    recs = []
    for name in ("meta.jsonl", "vectors.npy", "processed_files.json"):
        p = bdir / name
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        info = {"file": name, "exists": exists, "size": size}
        if exists and name == "vectors.npy":
            try:
                arr = np.load(p)
                info["shape"] = tuple(arr.shape)
            except Exception:
                info["shape"] = None
        recs.append(info)
    try:
        import pandas as pd
        return pd.DataFrame.from_records(recs)
    except Exception:
        return recs


def restore_from_backup_local(shard_dir: Path, bdir: Path):
    """
    バックアップから主要ファイルを復元（上書き）
    戻り値: (restored list, missing list)
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    restored, missing = [], []
    for name in ("meta.jsonl", "vectors.npy", "processed_files.json"):
        src = bdir / name
        dst = shard_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            restored.append(f"{name} <- {src.name}")
        else:
            missing.append(name)
    return restored, missing


def backup_age_days_local(backup_root: Path, backend: str, shard_id: str) -> Optional[float]:
    """
    直近（Rollback を除外）のバックアップからの経過日数を返す。
    無ければ None
    """
    dirs = list_backup_dirs_local(backup_root, backend, shard_id, include_rollback=False)
    if not dirs:
        return None
    latest = dirs[0]
    # フォルダ名の先頭 "YYYYMMDD-HHMMSS" から日時を読む（失敗しても None）
    try:
        ts = latest.name.split("-")[0] + "-" + latest.name.split("-")[1]
        dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
        now = datetime.now()
        return (now - dt).total_seconds() / 86400.0
    except Exception:
        return None


def cleanup_old_backups_keep_last(backup_root: Path, backend: str, shard_id: str, *, keep_last: int = 3) -> List[Path]:
    """
    最新 keep_last 件を残して、それより古いバックアップを削除（Rollback含む）
    """
    dirs = list_backup_dirs_local(backup_root, backend, shard_id, include_rollback=True)
    to_delete = dirs[keep_last:]
    deleted = []
    for d in to_delete:
        try:
            shutil.rmtree(d)
            deleted.append(d)
        except Exception:
            pass
    return deleted


def cleanup_old_backups_older_than_days(backup_root: Path, backend: str, shard_id: str, *, older_than_days: int = 90) -> List[Path]:
    """
    older_than_days より古いバックアップを削除（Rollback含む）
    """
    dirs = list_backup_dirs_local(backup_root, backend, shard_id, include_rollback=True)
    deleted = []
    for d in dirs:
        try:
            # "YYYYMMDD-HHMMSS" → 日数判定
            parts = d.name.split("-")
            ts = f"{parts[0]}-{parts[1]}"
            dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
            age_days = (datetime.now() - dt).days
            if age_days > older_than_days:
                shutil.rmtree(d)
                deleted.append(d)
        except Exception:
            # 解析できない名前は消さない
            pass
    return deleted
