# lib/backup_utils.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
import shutil
import pandas as pd
from typing import List, Tuple, Optional

def timestamp() -> str:
    """ローカルタイムの YYYYmmdd-HHMMSS を返す。"""
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")

def backup_dir_for(backup_root: Path, backend: str, shard_id: str, ts: Optional[str] = None) -> Path:
    """バックアップ保存先フォルダ（backup_root/backend/shard_id/timestamp）を返す。"""
    if ts is None:
        ts = timestamp()
    return backup_root / backend / shard_id / ts

def backup_all_local(src_dir: Path, backup_root: Path, backend: str, shard_id: str) -> Tuple[List[str], Path]:
    """
    src_dir（= VS_ROOT/backend/shard）から meta.jsonl / vectors.npy / processed_files.json を
    backup_root/backend/shard/<timestamp>/ にコピー。存在するものだけコピー。
    戻り値: (コピーしたファイル名リスト, 作成先ディレクトリ)
    """
    ts_dir = backup_dir_for(backup_root, backend, shard_id)
    ts_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, ts_dir / name)
            copied.append(name)
    return copied, ts_dir

def list_backup_dirs_local(backup_root: Path, backend: str, shard_id: str) -> List[Path]:
    """バックアップフォルダ一覧（新しい順）。"""
    root = backup_root / backend / shard_id
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)

def preview_backup_local(bdir: Path) -> pd.DataFrame:
    """バックアップ1個の中身サイズ一覧を DataFrame で返す。"""
    rows = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        p = bdir / name
        if p.exists():
            size = p.stat().st_size
            rows.append({"name": name, "size(bytes)": size, "path": str(p)})
    return pd.DataFrame(rows)

def restore_from_backup_local(dst_dir: Path, bdir: Path) -> Tuple[List[str], List[str]]:
    """バックアップから meta.jsonl / vectors.npy / processed_files.json を復元。"""
    restored, missing = [], []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        src = bdir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            restored.append(name)
        else:
            missing.append(name)
    return restored, missing

def backup_age_days_local(backup_root: Path, backend: str, shard_id: str) -> Optional[float]:
    """最新バックアップからの経過日数（最終更新基準）。バックアップ無しなら None。"""
    import time
    bdirs = list_backup_dirs_local(backup_root, backend, shard_id)
    if not bdirs:
        return None
    latest = bdirs[0]
    mtimes = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        p = latest / name
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    if not mtimes:
        mtimes.append(latest.stat().st_mtime)
    age_sec = max(time.time() - max(mtimes), 0.0)
    return age_sec / 86400.0

# ===== 古いバックアップ削除 =====

def cleanup_old_backups_keep_last(backup_root: Path, backend: str, shard_id: str, keep_last: int = 3) -> List[Path]:
    """
    最新 keep_last 件を残して古いバックアップを削除する。
    戻り値: 削除したフォルダのリスト
    """
    root = backup_root / backend / shard_id
    if not root.exists():
        return []
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    to_delete = dirs[keep_last:] if len(dirs) > keep_last else []
    deleted = []
    for d in to_delete:
        try:
            shutil.rmtree(d)
            deleted.append(d)
        except Exception as e:
            print(f"[cleanup_old_backups_keep_last] 削除失敗: {d} ({e})")
    return deleted

def cleanup_old_backups_older_than_days(backup_root: Path, backend: str, shard_id: str, older_than_days: int) -> List[Path]:
    """
    バックアップフォルダ名の timestamp（YYYYmmdd-HHMMSS）またはフォルダ/中身の mtime が
    指定日数より古いものを削除。
    戻り値: 削除したフォルダのリスト
    """
    threshold_dt = datetime.now(timezone.utc).astimezone() - timedelta(days=older_than_days)
    root = backup_root / backend / shard_id
    if not root.exists():
        return []
    deleted = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True):
        # 1) フォルダ名から日時を推定
        dt_from_name = None
        try:
            dt_from_name = datetime.strptime(d.name, "%Y%m%d-%H%M%S").astimezone()
        except Exception:
            pass

        if dt_from_name is not None:
            dt_candidate = dt_from_name
        else:
            # 2) フォルダ内の最新 mtime で判定
            latest_mtime = None
            for p in d.rglob("*"):
                try:
                    mt = p.stat().st_mtime
                except Exception:
                    continue
                if latest_mtime is None or mt > latest_mtime:
                    latest_mtime = mt
            if latest_mtime is None:
                try:
                    latest_mtime = d.stat().st_mtime
                except Exception:
                    latest_mtime = None
            if latest_mtime is None:
                continue
            dt_candidate = datetime.fromtimestamp(latest_mtime).astimezone()

        if dt_candidate < threshold_dt:
            try:
                shutil.rmtree(d)
                deleted.append(d)
            except Exception as e:
                print(f"[cleanup_old_backups_older_than_days] 削除失敗: {d} ({e})")
    return deleted
