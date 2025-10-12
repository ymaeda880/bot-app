# lib/diff_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Set, Dict
import json
import urllib.parse
import unicodedata
import hashlib
import numpy as np
import pandas as pd

# 既存のバックアップユーティリティ
from lib.backup_utils import backup_all_local, list_backup_dirs_local

# ============================================================
# 基本ユーティリティ
# ============================================================

def normalize_path(s: Optional[str]) -> Optional[str]:
    """URLデコード→NFKC正規化→区切り統一（\\→/）。非文字列は None を返す。"""
    if not isinstance(s, str):
        return None
    try:
        dec = urllib.parse.unquote(s)
        norm = unicodedata.normalize("NFKC", dec)
        return norm.replace("\\", "/")
    except Exception:
        return s

def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(chunk_size), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None

def vector_dim(vec_path: Path) -> Optional[Tuple[int, int]]:
    """vectors.npy の shape を (n, d) で返す。無い/壊れは None。"""
    if not vec_path.exists():
        return None
    try:
        arr = np.load(vec_path)
        return tuple(arr.shape)  # (n, d)
    except Exception:
        return None

# ============================================================
# meta / vectors / processed の統計（差分サマリ用）
# ============================================================

def meta_stats(p: Path) -> dict:
    if not p.exists():
        return {"exists": False, "size": 0, "md5": None, "rows": None}
    rows = 0
    try:
        with p.open("r", encoding="utf-8") as f:
            for _ in f:
                rows += 1
    except Exception:
        rows = None
    return {"exists": True, "size": p.stat().st_size, "md5": md5_file(p), "rows": rows}

def vectors_stats(p: Path) -> dict:
    if not p.exists():
        return {"exists": False, "size": 0, "md5": None, "shape": None}
    shape = None
    try:
        arr = np.load(p)
        shape = tuple(arr.shape)
    except Exception:
        shape = None
    return {"exists": True, "size": p.stat().st_size, "md5": md5_file(p), "shape": shape}

def processed_stats(p: Path) -> dict:
    if not p.exists():
        return {"exists": False, "size": 0, "md5": None, "count": None}
    count = None
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            count = len(obj)
        elif isinstance(obj, dict):
            count = len(obj.keys())
    except Exception:
        count = None
    return {"exists": True, "size": p.stat().st_size, "md5": md5_file(p), "count": count}

def compare_item(backend: str, shard: str, item: str, live_path: Path, backup_path: Path) -> dict:
    if item == "meta.jsonl":
        live = meta_stats(live_path)
        bak  = meta_stats(backup_path)
        keys = ("exists", "size", "md5", "rows")
    elif item == "vectors.npy":
        live = vectors_stats(live_path)
        bak  = vectors_stats(backup_path)
        keys = ("exists", "size", "md5", "shape")
    else:  # processed_files.json
        live = processed_stats(live_path)
        bak  = processed_stats(backup_path)
        keys = ("exists", "size", "md5", "count")

    different = any(live.get(k) != bak.get(k) for k in keys)
    return {
        "backend": backend,
        "shard_id": shard,
        "item": item,
        "different": different,
        "live_exists": live.get("exists"),
        "bak_exists": bak.get("exists"),
        "live_size": live.get("size"),
        "bak_size": bak.get("size"),
        "live_md5": live.get("md5"),
        "bak_md5": bak.get("md5"),
        "live_rows": live.get("rows") if "rows" in live else None,
        "bak_rows": bak.get("rows") if "rows" in bak else None,
        "live_shape": live.get("shape") if "shape" in live else None,
        "bak_shape": bak.get("shape") if "shape" in bak else None,
        "live_count": live.get("count") if "count" in live else None,
        "bak_count": bak.get("count") if "count" in bak else None,
    }

# ============================================================
# meta.jsonl の読み取り
# ============================================================

def read_meta_pairs(meta_file: Path, drop_empty: bool = False) -> Set[Tuple[Optional[str], Optional[str]]]:
    """
    meta.jsonl → {(pno, file)} 集合。
    - file/pno は normalize（URLデコード, NFKC, 区切り統一）。pno は str 化。
    - drop_empty=False（既定）: 空(None,"")でも集合に含める（差分が消えてしまうのを防ぐ）
    """
    pairs: Set[Tuple[Optional[str], Optional[str]]] = set()
    if not meta_file.exists():
        return pairs
    with meta_file.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            pno = obj.get("pno")
            pno = str(pno) if pno is not None else None
            file_ = normalize_path(obj.get("file"))
            if drop_empty and not file_:
                continue
            pairs.add((pno, file_))
    return pairs

def read_meta_dict(meta_file: Path) -> Dict[Tuple[Optional[str], Optional[str]], str]:
    """
    key=(pno, file_norm), val=1行JSON文字列(改行付き)
    """
    d: Dict[Tuple[Optional[str], Optional[str]], str] = {}
    if not meta_file.exists():
        return d
    with meta_file.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            pno = obj.get("pno")
            pno = str(pno) if pno is not None else None
            file_ = normalize_path(obj.get("file"))
            d[(pno, file_)] = json.dumps(obj, ensure_ascii=False) + "\n"
    return d

# ============================================================
# 差分集計 API（ページから呼び出し）
# ============================================================

def diff_all_shards(
    vs_root: Path,
    backup_root: Path,
    backend: str,
    shard_ids: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    各 shard について最新バックアップ（timestamp 配下）と現行の
    meta.jsonl / vectors.npy / processed_files.json を比較し、行サマリを返す。
    """
    records = []
    missing_backup: List[str] = []

    for sid in shard_ids:
        live_dir = vs_root / backend / sid
        live_meta = live_dir / "meta.jsonl"
        live_vec  = live_dir / "vectors.npy"
        live_pf   = live_dir / "processed_files.json"

        bdirs_sid = list_backup_dirs_local(backup_root, backend, sid)
        if not bdirs_sid:
            missing_backup.append(sid)
            # backup 不在の行も記録（backup_path はダミー）
            null_path = Path("/dev/null")
            records.append(compare_item(backend, sid, "meta.jsonl", live_meta, null_path))
            records.append(compare_item(backend, sid, "vectors.npy", live_vec, null_path))
            records.append(compare_item(backend, sid, "processed_files.json", live_pf, null_path))
            continue

        latest_bdir = bdirs_sid[0]
        bak_meta = latest_bdir / "meta.jsonl"
        bak_vec  = latest_bdir / "vectors.npy"
        bak_pf   = latest_bdir / "processed_files.json"

        records.append(compare_item(backend, sid, "meta.jsonl", live_meta, bak_meta))
        records.append(compare_item(backend, sid, "vectors.npy", live_vec, bak_vec))
        records.append(compare_item(backend, sid, "processed_files.json", live_pf, bak_pf))

    if not records:
        return pd.DataFrame(), missing_backup

    cols_view = [
        "backend", "shard_id", "item", "different",
        "live_exists", "bak_exists",
        "live_size", "bak_size",
        "live_rows", "bak_rows",
        "live_shape", "bak_shape",
        "live_count", "bak_count",
        "live_md5", "bak_md5",
    ]
    df = pd.DataFrame.from_records(records)
    df = df[cols_view]
    return df, missing_backup

def meta_diff_for_shard(
    vs_root: Path,
    backup_root: Path,
    backend: str,
    shard_id: str,
) -> Tuple[List[Tuple[Optional[str], Optional[str]]],
           List[Tuple[Optional[str], Optional[str]]],
           Optional[Path]]:
    """
    現行 vs 最新バックアップ の (pno, file) 差分。
    戻り値: (only_live_sorted, only_bak_sorted, latest_backup_dir or None)
    """
    bdirs_sid = list_backup_dirs_local(backup_root, backend, shard_id)
    if not bdirs_sid:
        return [], [], None

    latest_bdir = bdirs_sid[0]
    live_dir = vs_root / backend / shard_id
    live_meta = live_dir / "meta.jsonl"
    bak_meta  = latest_bdir / "meta.jsonl"

    live_pairs = read_meta_pairs(live_meta, drop_empty=False)
    bak_pairs  = read_meta_pairs(bak_meta,  drop_empty=False)

    only_live = sorted(list(live_pairs - bak_pairs))
    only_bak  = sorted(list(bak_pairs - live_pairs))
    return only_live, only_bak, latest_bdir

def load_only_bak_pairs_for_shard(
    vs_root: Path,
    backup_root: Path,
    backend: str,
    shard_id: str,
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], Optional[Path], int]:
    """
    バックアップ最新にのみ存在する差分（only_bak）を返す。
    ついでに only_live 件数も返す（UI の注意喚起用）。
    """
    only_live, only_bak, latest_bdir = meta_diff_for_shard(
        vs_root, backup_root, backend, shard_id
    )
    return only_bak, latest_bdir, len(only_live)

# ============================================================
# 🔄 差分同期（バックアップ → 現行）
#   - 指定された (pno, file) のみを meta.jsonl / vectors.npy / processed_files.json に反映
#   - 同期前に現行側の直前バックアップを作成（ロールバック用）
#   - 現行シャードが無い場合はフォルダと空ファイルを作成してから同期
# ============================================================

def sync_pairs_from_backup_to_live(
    VS_ROOT: Path,
    BACKUP_ROOT: Path,
    backend: str,
    shard_id: str,
    pairs_to_add: List[Tuple[Optional[str], Optional[str]]],
) -> dict:
    """
    指定 (pno,file) だけを バックアップ → 現行 に同期。
    returns: {"added": 追加件数, "live_backup_dir": "<直前バックアップdir or None>"}
    """
    result = {"added": 0, "live_backup_dir": None}

    # --- 最新バックアップを厳密に取得（timestamp ディレクトリ配下） ---
    bdirs = list_backup_dirs_local(BACKUP_ROOT, backend, shard_id)
    if not bdirs:
        raise FileNotFoundError("このシャードのバックアップが見つかりません。")
    latest_bdir = bdirs[0]
    bak_meta = latest_bdir / "meta.jsonl"
    bak_vec  = latest_bdir / "vectors.npy"

    # --- 現行側パス ---
    live_dir = VS_ROOT / backend / shard_id
    live_meta = live_dir / "meta.jsonl"
    live_vec  = live_dir / "vectors.npy"
    live_pf   = live_dir / "processed_files.json"

    # --- 現行初期化（フォルダ&空ファイル作成） ---
    live_dir.mkdir(parents=True, exist_ok=True)
    if not live_meta.exists():
        live_meta.touch()
    if not live_pf.exists():
        with live_pf.open("w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    # --- バックアップ側の存在/形状チェック ---
    if not bak_meta.exists() or not bak_vec.exists():
        raise FileNotFoundError(f"バックアップが不完全です: {latest_bdir}")

    bak_arr = np.load(bak_vec)
    if bak_arr.ndim != 2:
        raise ValueError("バックアップ側 vectors.npy の形状が不正です。")
    d = bak_arr.shape[1]

    # 現行 vectors が無い場合は空から
    if live_vec.exists():
        live_arr = np.load(live_vec)
        if live_arr.ndim != 2 or live_arr.shape[1] != d:
            raise ValueError("ベクトル次元が一致しません。")
    else:
        live_arr = np.empty((0, d))

    # --- 直前バックアップ（ロールバック用） ---
    try:
        _copied, bdir_live_backup = backup_all_local(live_dir, BACKUP_ROOT, backend, shard_id, label="Rollback")
        result["live_backup_dir"] = str(bdir_live_backup)
    except Exception:
        bdir_live_backup = None  # 失敗しても同期は継続

    # --- backup 側の (pno,file) → 行テキスト / 行index を構築 ---
    meta_dict = read_meta_dict(bak_meta)
    idx_map: Dict[Tuple[Optional[str], Optional[str]], int] = {}
    with bak_meta.open("r", encoding="utf-8") as f:
        i = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            pno  = obj.get("pno")
            pno  = str(pno) if pno is not None else None
            file_ = normalize_path(obj.get("file"))
            idx_map[(pno, file_)] = i
            i += 1

    # --- 追加対象を正規化して突き合わせ ---
    add_lines: List[str] = []
    add_indices: List[int] = []
    added_pairs: List[Tuple[Optional[str], Optional[str]]] = []

    for pno_in, file_in in pairs_to_add:
        pno_norm  = str(pno_in) if pno_in is not None else None
        file_norm = normalize_path(file_in)
        key = (pno_norm, file_norm)
        line = meta_dict.get(key)
        if line is None:
            # UIとの正規化差異でヒットしない可能性を最小化するため、さらにフォールバック探し
            # （file が空/None のケースもあるので、pno だけ一致で拾う等は危険。ここではスキップ）
            continue
        add_lines.append(line)
        if key in idx_map:
            add_indices.append(idx_map[key])
        added_pairs.append(key)

    if not add_lines:
        # 同期対象なし
        return result

    # --- meta.jsonl へ追記 ---
    with live_meta.open("a", encoding="utf-8") as f:
        f.writelines(add_lines)

    # --- vectors を結合保存 ---
    if add_indices:
        add_vecs = bak_arr[add_indices]
        new_vecs = np.vstack([live_arr, add_vecs])
        np.save(live_vec, new_vecs)
    else:
        # 念のため live をそのまま保存（存在しない場合は空を保存）
        np.save(live_vec, live_arr)

    # --- processed_files.json をユニーク追記 ---
    try:
        with live_pf.open("r", encoding="utf-8") as f:
            pf = json.load(f)
        if not isinstance(pf, list):
            pf = []
    except Exception:
        pf = []

    existing = set(normalize_path(x) for x in pf if isinstance(x, str))
    for _pno, fpath in added_pairs:
        norm = normalize_path(fpath) if isinstance(fpath, str) else None
        if norm and norm not in existing:
            pf.append(norm)
            existing.add(norm)

    with live_pf.open("w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)

    # 変更後（抜粋）
    result["added"] = len(added_pairs)

    # ★ 同期が正常終了したら「現在の状態」を PostOp で撮る
    try:
        _copied2, bdir_postop = backup_all_local(
            live_dir, BACKUP_ROOT, backend, shard_id, label="PostOp"
        )
        result["postop_backup_dir"] = str(bdir_postop)
    except Exception:
        result["postop_backup_dir"] = None

    return result
