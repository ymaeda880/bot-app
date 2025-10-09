# pages/51_メタファイルバックアップ（削除）.py
# ------------------------------------------------------------
# 🗑️ メタファイル削除ページ
# - 削除 / 初期化 / バックアップ / 復元 / 古いバックアップ整理
# - バックアップ保存先：<選択した BACKUP_ROOT> / <backend> / <shard_id> / <timestamp>
# - 追加機能:
#   1) すべてのシャードを即時バックアップ
#   2) 対象シャードのみ即時バックアップ
#   3) 「未バックアップ日数」しきい値で一括バックアップ
#   4) シャードごと削除（フォルダ完全削除→空フォルダ再作成）
#   5) 完全初期化にもバックアップ＆DELETE確認
#   6) 古いバックアップの削除（最新N件 / しきい値日数）
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import os
import urllib.parse
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

from config.path_config import PATHS  # ✅ vs_root / backup_root を集中管理
from lib.vectorstore_utils import iter_jsonl  # 既存ユーティリティ

# 🔁 外部ユーティリティ（lib 下へ切り出し済み）
from lib.backup_utils import (
    backup_all_local,
    list_backup_dirs_local,
    preview_backup_local,
    restore_from_backup_local,
    backup_age_days_local,
    cleanup_old_backups_keep_last,
    cleanup_old_backups_older_than_days,
)
from lib.processed_files_utils import remove_from_processed_files_selective

# ============================================================
# 基本パス（config に合わせる）
# ============================================================
VS_ROOT: Path = PATHS.vs_root
# デフォルトは標準バックアップ（ラジオで切替）
BACKUP_ROOT: Path = PATHS.backup_root

# ============================================================
# 初期存在保証（初回でも失敗しないように）
# ============================================================
try:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
(VS_ROOT / "openai").mkdir(parents=True, exist_ok=True)
(VS_ROOT / "local").mkdir(parents=True, exist_ok=True)

# ============================================================
# UI 設定
# ============================================================
st.set_page_config(page_title="51 メタファイル削除", page_icon="🗑️", layout="wide")
st.title("🗑️ メタファイルバックアップ・削除（シャード単位）")

# 冒頭の環境サマリ（改行で見やすく）
st.markdown(
    f"""
**VectorStore:** `{VS_ROOT}`  
**標準バックアップ:** `{PATHS.backup_root}`  
**外付けSSDバックアップ:** `{PATHS.backup_root2}`
    """,
    unsafe_allow_html=True,
)

st.info("このページは **削除・初期化・バックアップ/復元** に特化しています。作業前に必ずバックアップを作成してください。")

# ============================================================
# サイドバー: シャード選択
# ============================================================
with st.sidebar:
    st.header("対象シャード")
    backend = st.radio("バックエンド", ["openai", "local"], index=0, horizontal=True, key="sb_backend")
    base_backend_dir = VS_ROOT / backend
    if not base_backend_dir.exists():
        st.error(f"{base_backend_dir} が見つかりません。先にベクトル化を実行してください。")
        st.stop()

    shard_dirs = sorted([p for p in base_backend_dir.iterdir() if p.is_dir()])
    shard_ids = [p.name for p in shard_dirs]
    shard_id = st.selectbox("対象シャード", shard_ids if shard_ids else ["(なし)"], key="sb_shard")
    if not shard_ids:
        st.error("シャードが存在しません。")
        st.stop()

# ============================================================
# 対象シャードのパス
# ============================================================
base_dir = base_backend_dir / shard_id
meta_path = base_dir / "meta.jsonl"
vec_path  = base_dir / "vectors.npy"
pf_path   = base_dir / "processed_files.json"

# ============================================================
# 🔁 バックアップ（拡張機能）
# ============================================================
st.subheader("🛡️ バックアップ（拡張）")

# === バックアップ保存先の選択（標準 or 外付けSSD） ===
dest_label = st.radio(
    "バックアップ保存先",
    ["標準（backup_root）", "外付けSSD（backup_root2）"],
    horizontal=True,
    key="bak_dest"
)
# 選択に応じて BACKUP_ROOT を切替
BACKUP_ROOT = PATHS.backup_root if "標準" in dest_label else PATHS.backup_root2
try:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
st.caption(f"現在のバックアップ保存先: `{BACKUP_ROOT}`")

col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("⚡ 対象シャードを即時バックアップ", width="stretch", key="bak_one"):
        copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, backend, shard_id)
        if copied:
            st.success(f"[{backend}/{shard_id}] をバックアップ: {bdir}")
        else:
            st.warning(f"[{backend}/{shard_id}] コピー対象がありません。保存先: {bdir}")

with col_b:
    if st.button("⚡ すべてのシャードを即時バックアップ", width="stretch", key="bak_all"):
        summary = []
        for sid in shard_ids:
            sdir = base_backend_dir / sid
            copied, bdir = backup_all_local(sdir, BACKUP_ROOT, backend, sid)
            summary.append((sid, len(copied), bdir))
        ok = [f"- {sid}: {n}項目 -> {bdir}" for sid, n, bdir in summary]
        st.success("即時バックアップ完了:\n" + "\n".join(ok))

with col_c:
    threshold = st.selectbox("未バックアップ日数 以上ならバックアップ", [1, 2, 3, 7, 14, 30], index=2, key="bak_thr")
    if st.button("🗓 条件バックアップを実行", width="stretch", key="bak_cond"):
        triggered, skipped = [], []
        for sid in shard_ids:
            age = backup_age_days_local(BACKUP_ROOT, backend, sid)
            if age is None or age >= float(threshold):
                sdir = base_backend_dir / sid
                copied, bdir = backup_all_local(sdir, BACKUP_ROOT, backend, sid)
                triggered.append((sid, age, len(copied), bdir))
            else:
                skipped.append((sid, age))
        msg = ""
        if triggered:
            msg += "バックアップ実行（閾値超過 or 未実施）:\n" + "\n".join(
                f"- {sid}: age={('None' if age is None else f'{age:.2f}d')} -> {n}項目 @ {bdir}"
                for sid, age, n, bdir in triggered
            )
        if skipped:
            if msg:
                msg += "\n\n"
            msg += "スキップ（閾値未満）:\n" + "\n".join(f"- {sid}: age={age:.2f}d" for sid, age in skipped)
        st.info(msg or "対象がありませんでした。")

st.divider()

# ============================================================
# 🧹 古いバックアップの整理（最新N件 / しきい値日数）
# ============================================================
st.subheader("🧹 古いバックアップの整理")

# ★ 対象スコープを選択
scope_label = st.radio(
    "削除対象スコープ",
    ["現在のシャードのみ", "全シャード（backend配下すべて）"],
    horizontal=True,
    key="cleanup_scope"
)


c1, c2 = st.columns(2)
with c1:
    keep_last = st.number_input("保持する最新バックアップ数", min_value=1, max_value=50, value=3, step=1, key="keep_last_bak")
    if st.button("🧹 最新N件を残して古いバックアップを削除", width="stretch", key="btn_cleanup_keep_last"):
        targets = shard_ids if "全シャード" in scope_label else [shard_id]
        all_deleted = []
        for sid in targets:
            deleted = cleanup_old_backups_keep_last(BACKUP_ROOT, backend, sid, keep_last=int(keep_last))
            all_deleted.extend(deleted)
        if all_deleted:
            st.success(f"以下の古いバックアップを削除しました ({len(all_deleted)} 件):\n" +
                       "\n".join(f"- {d}" for d in all_deleted))
        else:
            st.info("削除対象のバックアップはありませんでした。")


with c2:
    older_days = st.number_input("この日数より古いバックアップを削除", min_value=1, max_value=3650, value=90, step=1, key="older_days_bak")
    if st.button("🧹 しきい値日数より古いバックアップを削除", width="stretch", key="btn_cleanup_older_than"):
        targets = shard_ids if "全シャード" in scope_label else [shard_id]
        all_deleted = []
        for sid in targets:
            deleted = cleanup_old_backups_older_than_days(BACKUP_ROOT, backend, sid, older_than_days=int(older_days))
            all_deleted.extend(deleted)
        if all_deleted:
            st.success(f"以下の古いバックアップを削除しました ({len(all_deleted)} 件):\n" +
                       "\n".join(f"- {d}" for d in all_deleted))
        else:
            st.info("削除対象のバックアップはありませんでした。")


st.divider()

# ============================================================
# 📄 現状プレビュー
# ============================================================
st.subheader("📄 現状プレビュー")
rows = [dict(obj) for obj in iter_jsonl(meta_path)] if meta_path.exists() else []
if not rows:
    st.warning("このシャードには meta.jsonl が存在しないか、レコードがありません。")
else:
    df = pd.DataFrame(rows)
    if "file" not in df.columns:
        df["file"] = None
    st.caption(f"レコード数: {len(df):,}")
    st.dataframe(df.head(500), width="stretch", height=420)

st.divider()

# ============================================================
# 📦 バックアップ（個別プレビュー）
# ============================================================
st.subheader("📦 バックアップ（個別プレビュー）")
bdirs = list_backup_dirs_local(BACKUP_ROOT, backend, shard_id)
if bdirs:
    sel_bdir_prev = st.selectbox("バックアッププレビュー", bdirs, format_func=lambda p: p.name, key="prev_bdir")
    if sel_bdir_prev:
        st.dataframe(preview_backup_local(sel_bdir_prev), width="stretch", height=180)
else:
    st.caption("まだバックアップがありません。")

st.divider()

# ============================================================
# 🧹 選択ファイル削除（processed_files の扱いをラジオで選択）
# ============================================================
st.subheader("🧹 選択ファイル削除")
if rows:
    files = sorted(pd.Series([r.get("file") for r in rows if r.get("file")]).unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        target_files = st.multiselect("削除対象ファイル（year/file.pdf など）", files, key="sel_targets")
    with c2:
        pf_mode = st.radio(
            "processed_files.json の処理",
            ["選択ファイルを消す（既定）", "完全リセット（全削除）", "変更しない"],
            index=0,
            key="pf_mode",
        )
        confirm_del = st.checkbox("削除に同意します", key="confirm_selective")

    if st.button(
        "🧹 削除実行",
        type="primary",
        width="stretch",
        disabled=not (target_files and confirm_del),
        key="btn_selective_delete",
    ):
        try:
            # 直前バックアップ
            copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, backend, shard_id)

            # meta.jsonl 再構築 + vectors.npy 同期
            keep_lines, keep_vec_indices = [], []
            removed_meta, valid_idx = 0, 0
            target_set = set(target_files)

            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    for raw in f:
                        s = raw.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            keep_lines.append(raw)  # 壊れ行は保全
                            continue
                        fname = obj.get("file") if isinstance(obj, dict) else None
                        if fname in target_set:
                            removed_meta += 1
                            valid_idx += 1
                            continue
                        keep_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")
                        keep_vec_indices.append(valid_idx)
                        valid_idx += 1
                with meta_path.open("w", encoding="utf-8") as f:
                    f.writelines(keep_lines)

            removed_vecs = 0
            if vec_path.exists():
                vecs = np.load(vec_path)
                if vecs.ndim == 2:
                    new_vecs = vecs[keep_vec_indices] if keep_vec_indices else np.empty((0, vecs.shape[1]))
                    removed_vecs = vecs.shape[0] - new_vecs.shape[0]
                    np.save(vec_path, new_vecs)

            # processed_files.json の処理（ラジオ選択）
            pf_msg = ""
            if pf_mode == "完全リセット（全削除）":
                if pf_path.exists():
                    pf_path.unlink()
                pf_msg = "- processed_files.json: 削除（完全リセット）\n"

            elif pf_mode == "選択ファイルを消す（既定）":
                if pf_path.exists():
                    before, after, removed_pf, removed_list = remove_from_processed_files_selective(
                        pf_path, target_files
                    )
                    if removed_pf > 0:
                        st.success(
                            "processed_files.json を更新しました:\n"
                            f"- 除外数: {removed_pf} 件 (before={before}, after={after})\n"
                            f"- 除外された項目の例: {removed_list}"
                        )
                    else:
                        st.warning(
                            "processed_files.json に一致する項目が見つかりませんでした。\n"
                            "（フル/相対/basename/stem・NFKC・URLデコード・区切り統一で照合しています）"
                        )
                else:
                    pf_msg = "- processed_files.json: 見つかりませんでした（処理スキップ）\n"

            else:  # "変更しない"
                pf_msg = "- processed_files.json: 変更なし\n"

            st.success(
                "削除完了 ✅\n"
                f"- meta.jsonl: {removed_meta} 行削除\n"
                f"- vectors.npy: {removed_vecs} 行削除\n"
                f"{pf_msg}"
                f"- バックアップ: {bdir}"
            )
        except Exception as e:
            st.error(f"削除中にエラー: {e}")

st.divider()

# ============================================================
# 🗂️ シャードごと削除（フォルダ完全削除）
# ============================================================
st.subheader("🗂️ シャードごと削除（フォルダ完全削除）")

def _dir_stats(d: Path) -> tuple[int, int]:
    if not d.exists():
        return (0, 0)
    n, total = 0, 0
    for p in d.rglob("*"):
        if p.is_file():
            n += 1
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return n, total

if base_dir.exists():
    cnt, total = _dir_stats(base_dir)
    st.caption(f"このシャードフォルダ配下のファイル数: **{cnt:,}** / 合計サイズ: **{total:,} bytes**")
else:
    st.caption("このシャードフォルダは存在しません。")

colx, coly = st.columns([2, 1])
with colx:
    do_backup_before_shard_delete = st.checkbox(
        "削除前に標準バックアップ（meta/vectors/processed）を作成する",
        value=True,
        key="sharddel_backup",
    )
    confirm_shard_del = st.checkbox("シャードごと削除に同意します（元に戻せません）", key="sharddel_confirm")
with coly:
    typed = st.text_input("タイプ確認：DELETE と入力", value="", key="sharddel_typed")

if st.button(
    "🗂️ シャードごと削除を実行",
    type="secondary",
    width="stretch",
    disabled=not (confirm_shard_del and typed.strip().upper() == "DELETE"),
    key="sharddel_exec",
):
    try:
        if do_backup_before_shard_delete and base_dir.exists():
            copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, backend, shard_id)
            st.info(f"事前バックアップ: {bdir} / コピー: {', '.join(copied) if copied else 'なし'}")

        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        st.success(f"シャード `{backend}/{shard_id}` を削除（フォルダ再作成済み）")
    except Exception as e:
        st.error(f"シャード削除中にエラー: {e}")

st.divider()

# ============================================================
# 🎯 year/pno で削除
# ============================================================
st.subheader("🎯 year / pno で削除")

if rows:
    years = sorted(set(str(r.get("year")) for r in rows if r.get("year")))
    pnos = sorted(set(str(r.get("pno")) for r in rows if r.get("pno")))

    c1, c2 = st.columns(2)
    with c1:
        sel_year = st.selectbox("対象 year", ["(未選択)"] + years, key="sel_year")
    with c2:
        sel_pno = st.selectbox("対象 pno", ["(未選択)"] + pnos, key="sel_pno")

    confirm_yp = st.checkbox("削除に同意します（バックアップ推奨）", key="confirm_yearpno")

    if st.button("🧹 year/pno 指定削除を実行", type="primary", width="stretch", disabled=not confirm_yp, key="btn_del_yearpno"):
        try:
            # バックアップ
            copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, backend, shard_id)

            keep_lines, keep_vec_indices = [], []
            removed_meta, valid_idx = 0, 0

            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    for raw in f:
                        s = raw.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            keep_lines.append(raw)
                            continue

                        yr = str(obj.get("year", ""))
                        pno = str(obj.get("pno", ""))
                        # ★ year/pno が一致するレコードを削除
                        if (sel_year != "(未選択)" and yr != sel_year):
                            keep_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")
                            keep_vec_indices.append(valid_idx)
                        elif (sel_pno != "(未選択)" and pno != sel_pno):
                            keep_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")
                            keep_vec_indices.append(valid_idx)
                        else:
                            removed_meta += 1
                        valid_idx += 1

                with meta_path.open("w", encoding="utf-8") as f:
                    f.writelines(keep_lines)

            # vectors.npy 同期
            removed_vecs = 0
            if vec_path.exists():
                vecs = np.load(vec_path)
                if vecs.ndim == 2:
                    new_vecs = vecs[keep_vec_indices] if keep_vec_indices else np.empty((0, vecs.shape[1]))
                    removed_vecs = vecs.shape[0] - new_vecs.shape[0]
                    np.save(vec_path, new_vecs)

            st.success(
                f"削除完了 ✅\n"
                f"- year={sel_year}, pno={sel_pno} に一致するメタを削除\n"
                f"- meta.jsonl: {removed_meta} 行削除\n"
                f"- vectors.npy: {removed_vecs} 行削除\n"
                f"- バックアップ: {bdir}"
            )
        except Exception as e:
            st.error(f"削除中にエラー: {e}")

st.divider()

# ============================================================
# 🗑️ 完全初期化（3ファイルのみ削除：meta.jsonl / vectors.npy / processed_files.json）
# ============================================================
st.subheader("🗑️ 完全初期化")

targets = [
    ("meta.jsonl", meta_path),
    ("vectors.npy", vec_path),
    ("processed_files.json", pf_path),
]
present = [(name, p, (p.stat().st_size if p.exists() and p.is_file() else 0)) for name, p in targets if p.exists()]
total_bytes = sum(s for _, _, s in present)

if present:
    lines = [f"- {name}: {p} ({size:,} bytes)" for name, p, size in present]
    st.caption("削除対象（存在しているもののみ）:\n" + "\n".join(lines))
    st.caption(f"合計サイズ: **{total_bytes:,} bytes**")
else:
    st.caption("削除対象のファイルは見つかりませんでした（meta / vectors / processed）。")

col_init_l, col_init_r = st.columns([2, 1])
with col_init_l:
    do_backup_before_wipe = st.checkbox(
        "削除前に標準バックアップ（meta/vectors/processed）を作成する",
        value=True,
        key="wipe_backup",
    )
    confirm_wipe = st.checkbox("完全初期化に同意します（元に戻せません）", key="wipe_confirm")
with col_init_r:
    typed_init = st.text_input("タイプ確認：DELETE と入力", value="", key="wipe_typed")

if st.button(
    "🗑️ 初期化実行",
    type="secondary",
    width="stretch",
    disabled=not (confirm_wipe and typed_init.strip().upper() == "DELETE"),
    key="wipe_execute",
):
    try:
        if do_backup_before_wipe:
            copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, backend, shard_id)
            st.info(f"事前バックアップ: {bdir} / コピー: {', '.join(copied) if copied else 'なし'}")

        deleted = []
        for name, p in targets:
            if p.exists():
                p.unlink()
                deleted.append(f"{name}: {p}")

        if deleted:
            st.success("完全初期化しました:\n" + "\n".join(f"- {x}" for x in deleted))
        else:
            st.info("削除対象のファイルがありませんでした。")
    except Exception as e:
        st.error(f"完全初期化中にエラー: {e}")

st.divider()

# ============================================================
# ♻️ バックアップ復元
# ============================================================
st.subheader("♻️ バックアップ復元")
bdirs = list_backup_dirs_local(BACKUP_ROOT, backend, shard_id)
if not bdirs:
    st.info("バックアップがありません。先に『バックアップ作成』を実行してください。")
else:
    sel_bdir_restore = st.selectbox("復元するバックアップを選択", bdirs, format_func=lambda p: p.name, key="restore_bdir")
    if sel_bdir_restore:
        st.dataframe(preview_backup_local(sel_bdir_restore), width="stretch", height=160)
        ok_restore = st.checkbox("復元に同意します（現在のファイルは上書きされます）", key="restore_ok")
        if st.button("♻️ 復元実行", type="primary", width="stretch", disabled=not ok_restore, key="restore_exec"):
            try:
                restored, missing = restore_from_backup_local(base_dir, sel_bdir_restore)
                msg = "復元完了 ✅\n" + "\n".join(f"- {x}" for x in restored)
                if missing:
                    msg += "\n\n存在しなかったバックアップ項目:\n" + "\n".join(f"- {x}" for x in missing)
                st.success(msg)
            except Exception as e:
                st.error(f"復元中にエラー: {e}")
