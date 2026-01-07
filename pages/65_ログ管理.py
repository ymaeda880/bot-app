# -*- coding: utf-8 -*-
# pages/65_ログ管理.py
# ============================================================
# 📜 ボットログ管理ビューア（管理者専用）
#  - 管理者監査ログ（JSONL）: Storages/logs/bot_app/ 配下（bot_YYYY-MM.jsonl 等）
#  - ユーザー別に一覧表示＋CSV出力
#  - ✅ 利用者履歴（SQLite）はここでは扱わない（別ページで実装）
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path
import datetime as dt

import streamlit as st
import pandas as pd

# ============================================================
# sys.path（設計上の確定事項：bot_project/bot_app から PROJECTS_ROOT を決め打ち）
# ============================================================
_THIS = Path(__file__).resolve()
APP_DIR = _THIS.parents[1]      # .../bot_app
PROJ_DIR = _THIS.parents[2]     # .../bot_project
PROJECTS_ROOT = PROJ_DIR.parent # .../projects

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

# ============================================================
# imports（common_lib に寄せる）
# ============================================================
from common_lib.auth.auth_helpers import (
    get_current_user_from_session_or_cookie,
    require_admin_user,
)

from common_lib.storage.external_ssd_root import resolve_storage_subdir_root

from lib.logs.log_io import load_jsonl_logs
from lib.logs.log_filters import calc_date_range, apply_common_filters


# ============================================================
# Streamlit 基本設定
# ============================================================
st.set_page_config(
    page_title="ボットログ管理（管理者専用）",
    page_icon="📜",
    layout="wide",
)

JST = dt.timezone(dt.timedelta(hours=9), name="Asia/Tokyo")
APP_NAME = "bot_app"


# ============================================================
# アクセス制御（管理者のみ：UIは page 側で出す）
# ============================================================
admin_user = require_admin_user(st)
if not admin_user:
    st.error("🚫 このページは管理者のみアクセスできます。")
    st.stop()


# ============================================================
# タイトル＋ログインバッジ（UIは page 側）
# ============================================================
left, right = st.columns([5, 2], vertical_alignment="center")
with left:
    st.title("📜 ボットログ管理（管理者専用）")
with right:
    st.success(f"✅ 管理者ログイン中: **{admin_user}**")


# ============================================================
# Storages ルート解決（settings.toml は使わない）
# ============================================================
STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)

# 管理者監査ログ：Storages/logs/bot_app/
ADMIN_LOGS_DIR = (STORAGE_ROOT / "logs" / APP_NAME).resolve()



# ============================================================
# JSONL候補列挙（bot_YYYY-MM.jsonl を想定）
# ============================================================
def list_bot_jsonl_candidates(admin_logs_dir: Path) -> list[Path]:
    """
    Storages/logs/bot_app/ 配下から JSONL を候補として列挙する。

    想定:
      - bot_app_YYYY-MM.jsonl
      - bot_app_YYYY-MM.log.jsonl（残っていても拾う）
      - bot_app.jsonl（rotate="none" 旧式：必要なら拾うが、通常は除外してよい）
    """
    if not admin_logs_dir.exists():
        return []

    cands: list[Path] = []
    cands += sorted(admin_logs_dir.glob("bot_app_*.jsonl"))
    cands += sorted(admin_logs_dir.glob("bot_app_*.log.jsonl"))

    # bot_app.jsonl（単一ログ）を “混在表示したい” 場合だけ拾う（通常は不要）
    single = admin_logs_dir / "bot_app.jsonl"
    if single.exists():
        cands.append(single)

    # 重複排除（順序維持）
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in cands:
        rp = str(p.resolve())
        if Path(rp).exists() and rp not in seen:
            uniq.append(Path(rp))
            seen.add(rp)

    # なるべく新しい月が上（ファイル名降順）
    return sorted(uniq, key=lambda x: x.name, reverse=True)



jsonl_files = list_bot_jsonl_candidates(ADMIN_LOGS_DIR)

# 月次ログ（bot_app_YYYY-MM*.jsonl）だけに絞りたい場合はここでフィルタ
# 例: bot_app.jsonl（単一ログ）を除外したいならコメントアウト解除
jsonl_files = [p for p in jsonl_files if p.name != "bot_app.jsonl"]

if not jsonl_files:
    st.error("\n".join([
        "管理者監査ログ（JSONL）が見つかりません。",
        f"期待ディレクトリ: {ADMIN_LOGS_DIR}",
        "例: Storages/logs/bot_app/bot_app_2026-01.jsonl",
    ]))
    st.stop()

# 全ファイルを対象（新しい順のまま）
BOT_LOG_FILES = [p.resolve() for p in jsonl_files]



# ============================================================
# ログファイル情報（表示）
# ============================================================
with st.expander("📁 ログファイル情報", expanded=True):
    st.write(f"**storages_root:** `{STORAGE_ROOT}`")
    st.write(f"**admin logs dir:** `{ADMIN_LOGS_DIR}`")
    st.write(f"**JSONL 対象ファイル数:** **{len(BOT_LOG_FILES)}**")

    # 最新更新（mtime最大）を表示
    try:
        newest = max(BOT_LOG_FILES, key=lambda p: p.stat().st_mtime)
        mtime = dt.datetime.fromtimestamp(newest.stat().st_mtime, tz=JST)
        st.write(f"**最新更新ファイル:** `{newest.name}`")
        st.write(f"**最終更新:** {mtime:%Y-%m-%d %H:%M:%S %Z}")
    except Exception:
        pass

    # ファイル一覧（多い場合もあるので折りたたみ）
    with st.expander("対象ファイル一覧（降順）", expanded=False):
        for p in BOT_LOG_FILES:
            st.write(f"- `{p.name}`")


# ============================================================
# ログ読み込み（JSONLのみ・キャッシュ）
# ============================================================
@st.cache_data(show_spinner=False)
def get_df_json_multi(paths: tuple[str, ...], mtimes: tuple[float, ...]) -> pd.DataFrame:
    # mtimes はキャッシュ破棄用（中では使わない）
    dfs: list[pd.DataFrame] = []
    for s in paths:
        p = Path(s)
        try:
            d = load_jsonl_logs(p)
            if d is not None and not d.empty:
                # どのファイル由来か追跡できるようにする（任意だが監査に便利）
                d = d.copy()
                d["__source_file"] = p.name
                dfs.append(d)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True, sort=False)
    return out


# キャッシュキー用（追記されたら mtime が変わる）
_paths = tuple(str(p) for p in BOT_LOG_FILES)
_mtimes = tuple(float(p.stat().st_mtime) for p in BOT_LOG_FILES)

df_json = get_df_json_multi(_paths, _mtimes)


if df_json.empty:
    st.warning("JSONLログが空です（またはまだ作成されていません）。ボット利用後に再度確認してください。")
    st.stop()


# ============================================================
# フィルタ UI（期間・アクション・ページ）
# ============================================================
st.divider()
st.subheader("🔍 フィルタ")

min_date, max_date = calc_date_range(df_json, pd.DataFrame())

actions: list[str] = []
if "action" in df_json.columns:
    actions = sorted([str(x) for x in df_json["action"].dropna().unique().tolist() if str(x).strip()])

pages: list[str] = []
if "page" in df_json.columns:
    pages = sorted([str(x) for x in df_json["page"].dropna().unique().tolist() if str(x).strip()])

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])

with c1:
    date_from = st.date_input("開始日", value=min_date or dt.date.today())
with c2:
    date_to = st.date_input("終了日", value=max_date or dt.date.today())
with c3:
    picked_actions = st.multiselect(
        "アクション（任意）",
        options=actions,
        default=actions,
        help="例: ask / answer など。未選択なら全種類。",
    )
with c4:
    picked_pages = st.multiselect(
        "ページ（任意）",
        options=pages,
        default=pages,
        help="同じログを共有している場合に利用。",
    )


# ============================================================
# JSONL フィルタ＆表示
# ============================================================
st.divider()
st.subheader("📘 JSONL（管理者監査ログ）")

fdf_json = apply_common_filters(
    df_json,
    date_from=date_from,
    date_to=date_to,
    actions=picked_actions,
    pages=picked_pages,
)

st.caption(f"対象: **{len(fdf_json):,} / {len(df_json):,}**")

if fdf_json.empty:
    st.info("指定条件に一致するログがありません。")
    st.stop()

# user → ts の順
sort_cols: list[str] = []
if "user" in fdf_json.columns:
    sort_cols.append("user")
if "ts" in fdf_json.columns:
    sort_cols.append("ts")

fdf_json_sorted = fdf_json.sort_values(sort_cols, ascending=[True] * len(sort_cols)) if sort_cols else fdf_json

st.dataframe(fdf_json_sorted, width="stretch")

csv_all_json = fdf_json_sorted.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ フィルタ済みCSV（ユーザー順・全項目）",
    data=csv_all_json,
    file_name="bot_admin_jsonl_all_by_user.csv",
    mime="text/csv",
)

st.subheader("📜 個別ユーザー（全項目）")
user_list = sorted([str(x) for x in fdf_json_sorted.get("user", pd.Series(dtype=str)).dropna().unique().tolist()])

target_user = st.selectbox(
    "ユーザー",
    options=["（未選択）"] + user_list,
    index=0,
)

if target_user != "（未選択）":
    udf = fdf_json_sorted[fdf_json_sorted["user"] == target_user]
    st.caption(f"`{target_user}`: **{len(udf):,}**")
    st.dataframe(udf, width="stretch")

    csv_user = udf.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        f"⬇️ `{target_user}` のCSV",
        data=csv_user,
        file_name=f"bot_admin_jsonl_{target_user}_full.csv",
        mime="text/csv",
    )

st.info(f"✅ 管理者 `{admin_user}` として監査ログ（JSONL）を閲覧中です。")
