# pages/51_データベースバックアップ・削除.py
# ------------------------------------------------------------
# 🗑️ メタファイル削除ページ
# - 削除 / 初期化 / バックアップ / 復元 / 古いバックアップ整理
# - バックアップ保存先：<選択した BACKUP_ROOT> / <vectorspace> / <shard_id> / <timestamp>
# - 追加機能:
#   1) すべてのシャードを即時バックアップ
#   2) 対象シャードのみ即時バックアップ
#   3) 「未バックアップ日数」しきい値で一括バックアップ
#   4) シャードごと削除（フォルダ完全削除→空フォルダ再作成）
#   5) 完全初期化にもバックアップ＆DELETE確認
#   6) 古いバックアップの削除（最新N件 / しきい値日数）
#   7) 最新バックアップ vs 現行 差分集計（全シャード）
#   8) (pno,file) 差分抽出 / 選択同期（バックアップ→現行）
# ------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import sys

_THIS = Path(__file__).resolve()
APP_ROOT = _THIS.parents[1]          # pages -> app root
PROJECTS_ROOT = _THIS.parents[3]     # auth_portal/pages -> projects/auth_portal
import sys
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))


import json
import shutil

import numpy as np
import pandas as pd
import streamlit as st

from config.path_config import PATHS  # ✅ vs_root / backup_root を集中管理
from lib.rag.vectorstore_utils import iter_jsonl  # 既存ユーティリティ

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

from lib.diff_utils import (
    diff_all_shards,
    meta_diff_for_shard,
    load_only_bak_pairs_for_shard,
    sync_pairs_from_backup_to_live,
    normalize_path,
)

# from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie
# from common_lib.auth.auth_helpers import (
#     get_current_user_from_session_or_cookie,
#     is_admin,
#     clear_auth_caches,
# )

from common_lib.auth.auth_helpers import require_admin_user

# ============================================================
# 基本パス
# ============================================================
VS_ROOT: Path = PATHS.vs_root

# ============================================================
# UI 設定
# ============================================================
st.set_page_config(page_title="51 データベースバックアップ・削除", page_icon="🗑️", layout="wide")


# ============================================================
# アクセス制御（管理者チェック）※ 65_ログ管理.py と同方式
# ============================================================
# clear_auth_caches()

# user, payload = get_current_user_from_session_or_cookie(st)
# if not user:
#     st.stop()

# if not is_admin(user):
#     st.error("🚫 このページは管理者のみアクセスできます。")
#     st.stop()

sub = require_admin_user(st)
if not sub:
    st.error("🚫 このページは管理者のみアクセスできます。")
    st.stop()

#st.success(f"✅ 管理者ログイン中: **{sub}**")  # ← 表示はここで自由に

user = sub
# ============================================================
# タイトル＋ログインバッジ（65_ログ管理.py と同スタイル）
# ============================================================
col_title, col_user = st.columns([5, 2], vertical_alignment="center")

with col_title:
    st.title("🗑️ データベースバックアップ・削除")
    st.write("（シャード単位）")
with col_user:
    # ここは 65 と同じ呼び方
    #user, payload = get_current_user_from_session_or_cookie(st)
    if user:
        st.success(f"管理者としてログイン中: **{user}**")
    else:
        st.warning("未ログイン") 

st.markdown(
    f"""
**VectorStore:** `{VS_ROOT}`  
**標準バックアップ:** `{PATHS.backup_root}`  
**外付けSSDバックアップ:** `{PATHS.backup_root2}`  
**外付けSSDバックアップ2:** `{PATHS.backup_root3}`
    """,
    unsafe_allow_html=True,
)

st.info("このページは **削除・初期化・バックアップ/復元・差分同期** に特化しています。作業前に必ずバックアップを作成してください。")

# ============================================================
# サイドバー: vectorspace/バックアップ先/シャード選択（選択変更で即再読込）
# ============================================================
BACKUP_ROOT = Path(st.session_state.get("CURRENT_BACKUP_ROOT", str(PATHS.backup_root)))
try:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# sidebar で作るので、後段でも参照できるように初期化
vectorspace: str = "openai"
shard_ident: str | None = None
shard_ids_all: list[str] = []
live_shards: list[str] = []
backup_shards: list[str] = []

with st.sidebar:
    st.header("対象と保存先")

    # ============================
    # ★ DB切替（vectorspace）
    # ============================
    st.subheader("📚 対象DB（vectorstore）")

    vectorspace_label = st.radio(
        "対象",
        ["report（openai）", "規定集（openai_sample）"],
        index=0,
        help="report: data/vectorstore/openai / 規定集: data/vectorstore/openai_sample",
        key="sb_vectorspace_label",
    )

    VECTORSPACE_MAP = {
        "report（openai）": "openai",
        "規定集（openai_sample）": "openai_sample",
    }
    vectorspace = VECTORSPACE_MAP[vectorspace_label]
    st.caption(f"現在: `{vectorspace}`")

    st.divider()

    # 2) バックアップ保存先
    dest_label = st.radio(
        "バックアップ保存先",
        ["標準（backup_root）", "外付けSSD（backup_root2）", "外付けSSD2（backup_root3）"],
        horizontal=True,
        key="bak_dest_sidebar",
    )

    def _resolve_backup_root(label: str) -> Path:
        if "SSD2" in label:
            return PATHS.backup_root3
        if "SSD" in label:
            return PATHS.backup_root2
        return PATHS.backup_root

    BACKUP_ROOT = _resolve_backup_root(dest_label)
    try:
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    st.session_state["CURRENT_BACKUP_ROOT"] = str(BACKUP_ROOT)

    # 3) シャード一覧（現行 + 選択中バックアップ先の和集合）
    base_vs_dir = VS_ROOT / vectorspace
    live_shards = []
    if base_vs_dir.exists():
        try:
            live_shards = sorted([p.name for p in base_vs_dir.iterdir() if p.is_dir()])
        except Exception:
            live_shards = []

    backup_vs_dir = BACKUP_ROOT / vectorspace
    backup_shards = []
    if backup_vs_dir.exists():
        try:
            backup_shards = sorted([p.name for p in backup_vs_dir.iterdir() if p.is_dir()])
        except Exception:
            backup_shards = []

    shard_ids_all = sorted(set(live_shards) | set(backup_shards))

    def _label_for(sid: str) -> str:
        badges: list[str] = []
        if sid in live_shards:
            badges.append("現行")
        if sid in backup_shards:
            if BACKUP_ROOT == PATHS.backup_root:
                badges.append("B1")
            elif BACKUP_ROOT == PATHS.backup_root2:
                badges.append("B2")
            else:
                badges.append("B3")
        suffix = f"（{'/'.join(badges)}）" if badges else ""
        return f"{sid}{suffix}"

    current_shard = st.session_state.get("sb_shard_value")

    if not shard_ids_all:
        st.warning("シャードが見つかりません（現行と選択中バックアップ先のいずれにも存在しません）。")
        st.session_state["sb_shard_value"] = None
        shard_id = None
    else:
        if current_shard not in shard_ids_all:
            current_shard = shard_ids_all[0]

        default_idx = shard_ids_all.index(current_shard)
        sel = st.selectbox(
            "対象シャード（カッコ内は存在場所: 現行=VS_ROOT, B1/B2/B3=選択中バックアップ先）",
            options=shard_ids_all,
            index=default_idx,
            format_func=_label_for,
            key="sb_shard_selectbox",
        )
        st.session_state["sb_shard_value"] = sel
        shard_id = sel

    shard_ident = shard_id

# ============================================================
# 対象シャードのパス（現行）
# ============================================================
if shard_ident is None:
    st.stop()

base_dir = VS_ROOT / vectorspace / shard_ident
meta_path = base_dir / "meta.jsonl"
vec_path = base_dir / "vectors.npy"
pf_path = base_dir / "processed_files.json"

# 初期存在保証（現行側は空でもよいが、フォルダは作っておく）
base_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 🛡️ バックアップ（拡張）
# ============================================================
st.subheader("🛡️ バックアップ（拡張）（バックアップの実行）")
st.caption(f"現在のバックアップ保存先: `{BACKUP_ROOT}`")

col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("⚡ 対象シャードを即時バックアップ", key="bak_one"):
        copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, vectorspace, shard_ident)
        if copied:
            st.success(f"[{vectorspace}/{shard_ident}] をバックアップ: {bdir}")
        else:
            st.info(f"[{vectorspace}/{shard_ident}] のコピー対象がありません（空シャードかもしれません）。 保存先: {bdir}")

with col_b:
    if st.button("⚡ すべてのシャードを即時バックアップ", key="bak_all"):
        summary = []
        for sid in shard_ids_all:
            sdir = VS_ROOT / vectorspace / sid
            sdir.mkdir(parents=True, exist_ok=True)  # 空でもOK
            copied, bdir = backup_all_local(sdir, BACKUP_ROOT, vectorspace, sid)
            summary.append((sid, len(copied), bdir))
        ok = [f"- {sid}: {n}項目 -> {bdir}" for sid, n, bdir in summary]
        st.success("即時バックアップ完了:\n" + ("\n".join(ok) if ok else "対象なし"))

with col_c:
    threshold = st.selectbox("未バックアップ日数 以上ならバックアップ", [1, 2, 3, 7, 14, 30], index=2, key="bak_thr")
    if st.button("🗓 条件バックアップを実行", key="bak_cond"):
        triggered, skipped = [], []
        for sid in shard_ids_all:
            age = backup_age_days_local(BACKUP_ROOT, vectorspace, sid)
            if age is None or age >= float(threshold):
                sdir = VS_ROOT / vectorspace / sid
                sdir.mkdir(parents=True, exist_ok=True)
                copied, bdir = backup_all_local(sdir, BACKUP_ROOT, vectorspace, sid)
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

scope_label = st.radio(
    "削除対象スコープ",
    ["現在のシャードのみ", "全シャード（vectorspace配下すべて）"],
    horizontal=True,
    key="cleanup_scope",
)

c1, c2 = st.columns(2)

with c1:
    keep_last = st.number_input("保持する最新バックアップ数", min_value=1, max_value=50, value=3, step=1, key="keep_last_bak")
    if st.button("🧹 最新N件を残して古いバックアップを削除", key="btn_cleanup_keep_last"):
        targets = shard_ids_all if "全シャード" in scope_label else [shard_ident]
        all_deleted = []
        for sid in targets:
            deleted = cleanup_old_backups_keep_last(BACKUP_ROOT, vectorspace, sid, keep_last=int(keep_last))
            all_deleted.extend(deleted)
        if all_deleted:
            st.success("以下の古いバックアップを削除しました:\n" + "\n".join(f"- {d}" for d in all_deleted))
        else:
            st.info("削除対象のバックアップはありませんでした。")

with c2:
    older_days = st.number_input("この日数より古いバックアップを削除", min_value=1, max_value=3650, value=90, step=1, key="older_days_bak")
    if st.button("🧹 しきい値日数より古いバックアップを削除", key="btn_cleanup_older_than"):
        targets = shard_ids_all if "全シャード" in scope_label else [shard_ident]
        all_deleted = []
        for sid in targets:
            deleted = cleanup_old_backups_older_than_days(BACKUP_ROOT, vectorspace, sid, older_than_days=int(older_days))
            all_deleted.extend(deleted)
        if all_deleted:
            st.success("以下の古いバックアップを削除しました:\n" + "\n".join(f"- {d}" for d in all_deleted))
        else:
            st.info("削除対象のバックアップはありませんでした。")

st.divider()

# ============================================================
# 📄 現状プレビュー（現行）
# ============================================================
st.subheader("📄 現状プレビュー（現行）（先頭500行表示）")
rows = [dict(obj) for obj in iter_jsonl(meta_path)] if meta_path.exists() else []
if not rows:
    st.caption("このシャードには meta.jsonl が存在しないか、レコードがありません。")
else:
    df = pd.DataFrame(rows)
    if "file" not in df.columns:
        df["file"] = None
    st.caption(f"レコード数: {len(df):,}")
    st.dataframe(df.head(500), height=420)

st.divider()

# ============================================================
# 📦 バックアップ（個別プレビュー）
# ============================================================
st.subheader("📦 バックアップ（個別プレビュー）（バックアップ状況の確認）")
bdirs_prev = list_backup_dirs_local(BACKUP_ROOT, vectorspace, shard_ident)
if bdirs_prev:
    sel_bdir_prev = st.selectbox("バックアッププレビュー", bdirs_prev, format_func=lambda p: p.name, key="prev_bdir")
    if sel_bdir_prev:
        st.dataframe(preview_backup_local(sel_bdir_prev), height=180)
else:
    st.caption("まだバックアップがありません。")

st.divider()

# ============================================================
# 🧹 選択ファイル削除（内部メモリ内）
# ============================================================
st.subheader("🧹 選択ファイル削除（内部メモリ内）")
st.info(
    "- 選択ファイルに対応する meta.jsonl の行・vectors.npy の対応ベクトル・processed_files.json の項目を安全に削除します。\n"
    "- 実行前に **現行のロールバック用バックアップ（-Rollback）** を自動作成します。削除後は必要に応じて **-PostOp** を手動/自動で取得してください。"
)

if rows:
    files = sorted(pd.Series([r.get("file") for r in rows if r.get("file")]).unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        target_files = st.multiselect("削除対象ファイル（year/file.pdf など）", files, key="sel_targets")
    with c2:
        st.caption("processed_files.json の処理: 選択ファイルを processed_files.json から削除（既定動作）")
        confirm_del = st.checkbox("削除に同意します", key="confirm_selective")

    if st.button(
        "🧹 削除実行",
        type="primary",
        disabled=not (target_files and confirm_del),
        key="btn_selective_delete",
    ):
        try:
            copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, vectorspace, shard_ident, label="Rollback")
            if copied:
                st.caption(f"ロールバック用バックアップ: {bdir}")

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
                            keep_lines.append(raw)
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

            pf_msg = ""
            if pf_path.exists():
                before, after, removed_pf, removed_list = remove_from_processed_files_selective(
                    pf_path, target_files
                )
                if removed_pf > 0:
                    pf_msg = (
                        "processed_files.json を更新しました:\n"
                        f"- 除外数: {removed_pf} 件 (before={before}, after={after})\n"
                        f"- 除外された項目の例: {removed_list}"
                    )
            else:
                pf_msg = "- processed_files.json: 見つかりませんでした（処理スキップ）\n"

            st.success(
                "削除完了 ✅\n"
                f"- meta.jsonl: {removed_meta} 行削除\n"
                f"- vectors.npy: {removed_vecs} 行削除\n"
                f"{pf_msg}"
                f"\n- ロールバック: {bdir}"
            )

            try:
                _, bdir_post = backup_all_local(base_dir, BACKUP_ROOT, vectorspace, shard_ident, label="PostOp")
                st.caption(f"処理後スナップショット（PostOp）: {bdir_post}")
            except Exception:
                st.warning("処理後バックアップ（PostOp）の作成に失敗しました。")

        except Exception as e:
            st.error(f"削除中にエラー: {e}")

st.divider()

# ============================================================
# 🗂️ シャードごと削除（フォルダ完全削除）
# ============================================================
st.subheader("🗂️ シャードごと削除（フォルダ完全削除）（内部メモリ内）")
st.info(
    "- 選択中のシャード（VS_ROOT/<vectorspace>/<shard_id>）フォルダを丸ごと削除し、空で再作成します。\n"
    "- 実行前に **-Rollback** を自動作成します。必要なら実行後に **-PostOp** を作成してください。"
)

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

colx, coly = st.columns([2, 1])
with colx:
    do_backup_before_shard_delete = st.checkbox(
        "削除前に標準バックアップ（-Rollback）を作成する",
        value=True,
        key="sharddel_backup",
    )
    confirm_shard_del = st.checkbox("シャードごと削除に同意します（元に戻せません）", key="sharddel_confirm")
with coly:
    typed = st.text_input("タイプ確認：DELETE と入力", value="", key="sharddel_typed")

if st.button(
    "🗂️ シャードごと削除を実行",
    type="secondary",
    disabled=not (confirm_shard_del and typed.strip().upper() == "DELETE"),
    key="sharddel_exec",
):
    try:
        if do_backup_before_shard_delete and base_dir.exists():
            copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, vectorspace, shard_ident, label="Rollback")
            st.caption(f"事前バックアップ（Rollback）: {bdir}")

        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        st.success(f"シャード `{vectorspace}/{shard_ident}` を削除（フォルダ再作成済み）")

        try:
            _, bdir_post = backup_all_local(base_dir, BACKUP_ROOT, vectorspace, shard_ident, label="PostOp")
            st.caption(f"処理後スナップショット（PostOp）: {bdir_post}")
        except Exception:
            pass
    except Exception as e:
        st.error(f"シャード削除中にエラー: {e}")

st.divider()

# ============================================================
# 🎯 year/pno 指定削除
# ============================================================
st.subheader("🎯 year / pno 指定削除")
st.info(
    "- 指定の (year, pno) に一致する meta.jsonl の行と vectors.npy の対応ベクトル、processed_files.json を整合させます。\n"
    "- 実行前に **-Rollback** を自動作成、完了後に **-PostOp** を作成します。"
)

if rows:
    years = sorted(set(str(r.get("year")) for r in rows if r.get("year")))
    pnos = sorted(set(str(r.get("pno")) for r in rows if r.get("pno")))

    c1, c2 = st.columns(2)
    with c1:
        sel_year = st.selectbox("対象 year", ["(未選択)"] + years, key="sel_year")
    with c2:
        sel_pno = st.selectbox("対象 pno", ["(未選択)"] + pnos, key="sel_pno")

    confirm_yp = st.checkbox("削除に同意します（バックアップ推奨）", key="confirm_yearpno")

    if st.button(
        "🧹 year/pno 指定削除を実行",
        type="primary",
        disabled=not confirm_yp,
        key="btn_del_yearpno",
    ):
        try:
            has_filter = (sel_year != "(未選択)" or sel_pno != "(未選択)")
            if not has_filter:
                st.warning("year または pno を選択してください（両方未選択は実行されません）。")
            else:
                copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, vectorspace, shard_ident, label="Rollback")
                if copied:
                    st.caption(f"ロールバック用バックアップ: {bdir}")

                keep_lines, keep_vec_indices = [], []
                removed_meta, valid_idx = 0, 0
                removed_files = set()

                if meta_path.exists():
                    with meta_path.open("r", encoding="utf-8") as f:
                        for raw in f:
                            s = raw.strip()
                            if not s:
                                continue
                            try:
                                obj = json.loads(s)
                            except Exception:
                                keep_lines.append(raw + ("\n" if not raw.endswith("\n") else ""))
                                continue

                            yr = str(obj.get("year", ""))
                            pno = str(obj.get("pno", ""))
                            match_year = (sel_year != "(未選択)" and yr == sel_year)
                            match_pno = (sel_pno != "(未選択)" and pno == sel_pno)

                            if (sel_year == "(未選択)" or match_year) and (sel_pno == "(未選択)" or match_pno):
                                removed_meta += 1
                                fpath = obj.get("file")
                                if isinstance(fpath, str) and fpath:
                                    removed_files.add(fpath)
                            else:
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

                pf_msg = ""
                if removed_files:
                    if pf_path.exists():
                        before, after, removed_pf, removed_list = remove_from_processed_files_selective(
                            pf_path, sorted(removed_files)
                        )
                        if removed_pf > 0:
                            pf_msg = (
                                "processed_files.json を更新しました:\n"
                                f"- 除外数: {removed_pf} 件 (before={before}, after={after})\n"
                                f"- 除外された項目の例: {removed_list}\n"
                            )
                    else:
                        pf_msg = "processed_files.json が見つからなかったため更新をスキップしました。\n"

                st.success(
                    "削除完了 ✅\n"
                    f"- year={sel_year}, pno={sel_pno} に一致するメタを削除\n"
                    f"- meta.jsonl: {removed_meta} 行削除\n"
                    f"- vectors.npy: {removed_vecs} 行削除\n"
                    f"{pf_msg}"
                    f"- ロールバック: {bdir}"
                )

                try:
                    _, bdir_post = backup_all_local(base_dir, BACKUP_ROOT, vectorspace, shard_ident, label="PostOp")
                    st.caption(f"処理後スナップショット（PostOp）: {bdir_post}")
                except Exception:
                    st.warning("処理後バックアップ（PostOp）の作成に失敗しました。")

        except Exception as e:
            st.error(f"削除中にエラー: {e}")

st.divider()

# ============================================================
# 🗑️ 完全初期化（3ファイルのみ削除）
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
        "削除前に標準バックアップ（-Rollback）を作成する",
        value=True,
        key="wipe_backup",
    )
    confirm_wipe = st.checkbox("完全初期化に同意します（元に戻せません）", key="wipe_confirm")
with col_init_r:
    typed_init = st.text_input("タイプ確認：DELETE と入力", value="", key="wipe_typed")

if st.button(
    "🗑️ 初期化実行",
    type="secondary",
    disabled=not (confirm_wipe and typed_init.strip().upper() == "DELETE"),
    key="wipe_execute",
):
    try:
        if do_backup_before_wipe:
            copied, bdir = backup_all_local(base_dir, BACKUP_ROOT, vectorspace, shard_ident, label="Rollback")
            st.caption(f"事前バックアップ（Rollback）: {bdir}")

        deleted = []
        for name, p in targets:
            if p.exists():
                p.unlink()
                deleted.append(f"{name}: {p}")

        if deleted:
            st.success("完全初期化しました:\n" + "\n".join(f"- {x}" for x in deleted))
        else:
            st.info("削除対象のファイルがありませんでした。")

        try:
            _, bdir_post = backup_all_local(base_dir, BACKUP_ROOT, vectorspace, shard_ident, label="PostOp")
            st.caption(f"処理後スナップショット（PostOp）: {bdir_post}")
        except Exception:
            pass

    except Exception as e:
        st.error(f"完全初期化中にエラー: {e}")

st.divider()

# ============================================================
# 🔍 最新バックアップ vs 現行DB 差分チェック（全シャード）
# ============================================================
st.subheader("🔍 最新バックアップと現行DBの差分チェック（全シャード）")
st.info(
    "- 選択中のバックアップ保存先にある**最新バックアップ**と、**現行DB**を全シャードについて比較します。\n"
    "- 比較対象: meta.jsonl / vectors.npy / processed_files.json（存在・サイズ・MD5・件数/shape など）"
)

if st.button("🧮 最新バックアップとの差分を集計（全シャード）", type="secondary", key="btn_diff_all"):
    try:
        targets = shard_ids_all
        df_diff, missing_backup = diff_all_shards(VS_ROOT, BACKUP_ROOT, vectorspace, targets)
        if df_diff.empty:
            st.info("対象シャードが見つかりませんでした。")
        else:
            diff_items = int(df_diff["different"].sum()) if "different" in df_diff.columns else 0
            st.write(f"**差分あり:** {diff_items} / {len(df_diff)} 項目")
            if missing_backup:
                st.warning("最新バックアップが見つからないシャード: " + ", ".join(missing_backup))
            st.dataframe(df_diff, height=400)
    except Exception as e:
        st.error(f"差分集計中にエラー: {e}")

st.divider()

# ============================================================
# 🔎 対象シャードの meta.jsonl 差分（pno, file）を抽出
# ============================================================
st.subheader("🔎 対象シャードのメタ差分（pno, file）")
st.info(
    "- 現在選択されているシャードについて、**最新バックアップ**と**現行DB**の `(pno, file)` 差分を抽出します。\n"
    "- 片方にしか無いレコード（追加/欠落）を中心に表示します。"
)

if st.button("📑 対象シャードの差分 (pno, file) を抽出", type="secondary", key="btn_meta_diff_one"):
    try:
        only_live, only_bak, latest_bdir = meta_diff_for_shard(VS_ROOT, BACKUP_ROOT, vectorspace, shard_ident)
        if latest_bdir is None:
            st.warning("このシャードに **最新バックアップが見つかりません**。先にバックアップを作成してください。")
        else:
            st.write(f"**シャード:** `{vectorspace}/{shard_ident}`")
            st.write(f"**最新バックアップ:** `{latest_bdir.name}`")
            st.write(f"- 現行のみ: {len(only_live)} 件 / バックアップのみ: {len(only_bak)} 件")

            colL, colR = st.columns(2)
            with colL:
                st.markdown("**🟢 現行にのみ存在**")
                if only_live:
                    st.dataframe(pd.DataFrame(only_live, columns=["pno", "file"]), height=240)
                else:
                    st.caption("差分なし")
            with colR:
                st.markdown("**🟠 バックアップにのみ存在**")
                if only_bak:
                    st.dataframe(pd.DataFrame(only_bak, columns=["pno", "file"]), height=240)
                else:
                    st.caption("差分なし")

            st.success("差分抽出 完了 ✅")
    except Exception as e:
        st.error(f"差分抽出中にエラー: {e}")

st.divider()

# ============================================================
# ✅ 差分を選択して同期（チェックボックス付き）/ 全差分一括同期
# ============================================================
st.subheader("✅ 差分を選択して同期（対象シャード）")
st.info(
    "- **バックアップにのみ存在**する `(pno, file)` を一覧表示し、選択分だけ現行へ同期します（バックアップ→現行）。\n"
    "- 実行時に **現行の直前バックアップ（-Rollback）** を自動取得し、同期後は `sync_pairs_from_backup_to_live()` が **-PostOp** を自動作成します。\n"
    "- ※ 現行にのみ存在（only_live）は警告のみで同期対象外です（片方向同期で安全に）。"
)

if st.button("📥 差分を読み込む（バックアップ最新 vs 現行）", type="secondary", key="btn_load_diffs_select"):
    try:
        only_bak_pairs, latest_bdir, only_live_cnt = load_only_bak_pairs_for_shard(VS_ROOT, BACKUP_ROOT, vectorspace, shard_ident)
        st.session_state["diff_pairs_only_bak"] = only_bak_pairs
        st.session_state["diff_latest_bdir"] = str(latest_bdir) if latest_bdir else ""
        if only_live_cnt > 0:
            st.warning(f"現行にのみ存在する差分 (only_live) が {only_live_cnt} 件あります。片方向同期の対象外です。")
        st.success(f"差分（バックアップにのみ存在）を読み込みました: {len(only_bak_pairs)} 件")
    except Exception as e:
        st.error(f"差分読み込み中にエラー: {e}")

diff_pairs = st.session_state.get("diff_pairs_only_bak")
latest_bdir_str = st.session_state.get("diff_latest_bdir")

if diff_pairs is not None and latest_bdir_str is not None:
    if latest_bdir_str:
        st.caption(f"バックアップ最新: `{Path(latest_bdir_str).name}` / 差分件数: {len(diff_pairs)}")

    df_view = pd.DataFrame(diff_pairs, columns=["pno", "file"])
    if not df_view.empty:
        df_view.insert(0, "sync", True)

        edited = st.data_editor(
            df_view,
            height=360,
            column_config={
                "sync": st.column_config.CheckboxColumn("同期", default=True),
                "pno": st.column_config.TextColumn("pno"),
                "file": st.column_config.TextColumn("file"),
            },
            disabled=["pno", "file"],
            key="diff_editor_only_bak",
        )

        colL, colR = st.columns(2)

        with colL:
            if st.button("🟢 選択分のみ同期", type="primary", key="btn_sync_selected"):
                try:
                    sel_pairs = [
                        (str(row["pno"]) if row["pno"] is not None else None, normalize_path(row["file"]))
                        for _, row in edited.iterrows() if row.get("sync")
                    ]
                    res = sync_pairs_from_backup_to_live(VS_ROOT, BACKUP_ROOT, vectorspace, shard_ident, sel_pairs)
                    st.success(
                        f"同期完了 ✅ 追加 {res.get('added', 0)} 件\n"
                        f"・直前バックアップ(Rollback): {res.get('live_backup_dir')}\n"
                        f"・処理後スナップショット(PostOp): {res.get('postop_backup_dir')}"
                    )
                except Exception as e:
                    st.error(f"同期中にエラー: {e}")

        with colR:
            if st.button("🟣 全ての差分を同期", type="secondary", key="btn_sync_all"):
                try:
                    res = sync_pairs_from_backup_to_live(VS_ROOT, BACKUP_ROOT, vectorspace, shard_ident, diff_pairs)
                    st.success(
                        f"同期完了 ✅ 追加 {res.get('added', 0)} 件\n"
                        f"・直前バックアップ(Rollback): {res.get('live_backup_dir')}\n"
                        f"・処理後スナップショット(PostOp): {res.get('postop_backup_dir')}"
                    )
                except Exception as e:
                    st.error(f"同期中にエラー: {e}")
    else:
        st.caption("差分はありません。")
else:
    st.caption("↑ まず『差分を読み込む』を押して、同期候補を表示してください。")

st.divider()

# ============================================================
# ♻️ バックアップ復元（選択したバックアップ → 現行）
# ============================================================
st.subheader("♻️ バックアップ復元（内部メモリ内に復元）")
st.info(
    "- `<backup_root>/<vectorspace>/<shard_id>/<timestamp-Location[-Rollback|-PostOp]>` から、"
    " meta.jsonl / vectors.npy / processed_files.json を現行へ復元します（上書き）。"
)

bdirs_restore = list_backup_dirs_local(BACKUP_ROOT, vectorspace, shard_ident)
if not bdirs_restore:
    st.caption("バックアップがありません。先に『バックアップ作成』を実行してください。")
else:
    sel_bdir_restore = st.selectbox("復元するバックアップを選択", bdirs_restore, format_func=lambda p: p.name, key="restore_bdir")
    if sel_bdir_restore:
        st.dataframe(preview_backup_local(sel_bdir_restore), height=160)
        ok_restore = st.checkbox("復元に同意します（現在のファイルは上書きされます）", key="restore_ok")
        if st.button("♻️ 復元実行", type="primary", disabled=not ok_restore, key="restore_exec"):
            try:
                restored, missing = restore_from_backup_local(base_dir, sel_bdir_restore)
                msg = "復元完了 ✅\n" + "\n".join(f"- {x}" for x in restored)
                if missing:
                    msg += "\n\nバックアップ内に存在しなかった項目:\n" + "\n".join(f"- {x}" for x in missing)
                st.success(msg)
            except Exception as e:
                st.error(f"復元中にエラー: {e}")
