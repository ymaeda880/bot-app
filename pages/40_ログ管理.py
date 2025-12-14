# pages/40_ログ管理.py
# ============================================================
# 📜 ボットログ管理ビューア（管理者専用）
#  - logs/ 配下の bot_YYYY-MM.jsonl 等（複数）を選択して読み込み
#  - SQLite(bot_logs.sqlite3) の logs テーブルも読み込み
#  - 存在するログ項目（列）をすべて含めた形でユーザー別に一覧表示＋CSV出力
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path
import datetime as dt

import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────
# common_lib へのパスを自動追加（99_画像ログ集計.py と同系統）
# ─────────────────────────────────────────────────────────────
def _add_commonlib_parent_to_syspath() -> str | None:
    """このファイルから上方向に辿って common_lib / COMMON_LIB を探し、
    見つかったディレクトリの親ディレクトリを sys.path に追加する。
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        for name in ("common_lib", "COMMON_LIB"):
            if (parent / name).is_dir():
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                return str(parent)
    return None


_add_commonlib_parent_to_syspath()

# ─────────────────────────────────────────────────────────────
# 認証関連のインポート
# ─────────────────────────────────────────────────────────────
from common_lib.auth.auth_helpers import (
    get_current_user_from_session_or_cookie,
    is_admin,
    _resolve_settings_path,
    get_admin_users,
    clear_auth_caches,
)

# ログ読み込み＆フィルタ共通関数
from lib.logs.log_io import load_jsonl_logs, load_sqlite_logs
from lib.logs.log_filters import calc_date_range, apply_common_filters

# ============================================================
# Streamlit 基本設定
# ============================================================
st.set_page_config(
    page_title="ボットログ管理（管理者専用）",
    page_icon="📜",
    layout="wide",
)

# ============================================================
# タイトル＋ログインバッジ（22_画像生成.py と同じスタイル）
# ============================================================
col_title, col_user = st.columns([5, 2], vertical_alignment="center")

with col_title:
    st.title("📜 ボットログ管理（管理者専用）")

with col_user:
    user, payload = get_current_user_from_session_or_cookie(st)
    if user:
        st.success(f"管理者としてログイン中: **{user}**")
    else:
        st.warning("未ログイン（Cookie 未検出）")

# ============================================================
# アクセス制御（管理者チェック）
# ============================================================
clear_auth_caches()

if not user:
    st.stop()

if not is_admin(user):
    st.error("🚫 このページは管理者のみアクセスできます。")
    st.stop()

with st.expander("🪶 認証デバッグ情報（任意で閉じておいてOK）", expanded=False):
    st.write("設定ファイル探索結果:", _resolve_settings_path())
    st.write("管理者一覧:", sorted(get_admin_users()))
    st.write("現在のユーザー:", user)

# ============================================================
# ログファイル情報（JSONL / SQLite）
# ============================================================
APP_DIR = Path(__file__).resolve().parents[1]
APP_NAME = APP_DIR.name

LOGS_DIR = (APP_DIR / "logs").resolve()

# 旧: 単一ファイル（従来互換）
LEGACY_BOT_LOG_FILE = (LOGS_DIR / f"{APP_NAME}.log.jsonl").resolve()

# SQLite
SQLITE_DB = (LOGS_DIR / "bot_logs.sqlite3").resolve()

JST = dt.timezone(dt.timedelta(hours=9), name="Asia/Tokyo")


def list_bot_jsonl_candidates(logs_dir: Path) -> list[Path]:
    """
    logs_dir 配下から「ボット用らしき JSONL」を候補として列挙する。

    想定:
      - 新方式: bot_YYYY-MM.jsonl
      - 新方式(別名): bot_YYYY-MM.log.jsonl などでも拾える
      - 互換: {APP_NAME}.log.jsonl（従来の単一ファイル）
      - 念のため: ファイル名に 'bot' を含む *.jsonl も拾う

    ※ 画像ログ等が同じ logs/ にある場合は、
       必要に応じて条件をさらに絞ってください。
    """
    cands: list[Path] = []

    # 明示パターン（優先）
    cands += sorted(logs_dir.glob("bot_*.jsonl"))
    cands += sorted(logs_dir.glob("bot_*.log.jsonl"))

    # 念のため「bot」を含む jsonl も拾う（取りこぼし防止）
    for p in sorted(logs_dir.glob("*.jsonl")):
        if "bot" in p.stem.lower():
            cands.append(p)

    # 従来互換（最後に追加）
    if LEGACY_BOT_LOG_FILE.exists():
        cands.append(LEGACY_BOT_LOG_FILE)

    # 重複排除（順序維持）
    uniq: list[Path] = []
    seen = set()
    for p in cands:
        rp = p.resolve()
        if rp not in seen and rp.exists():
            uniq.append(rp)
            seen.add(rp)

    # なるべく新しい月が上に来るよう「ファイル名」で降順ソート
    # bot_YYYY-MM.jsonl の形式なら文字列ソートでだいたいOK
    uniq_sorted = sorted(uniq, key=lambda x: x.name, reverse=True)
    return uniq_sorted


# ============================================================
# サイドバー（JSONLファイル選択 + 更新）
# ============================================================
jsonl_files = list_bot_jsonl_candidates(LOGS_DIR)

with st.sidebar:
    st.header("ログ設定")

    if jsonl_files:
        def _fmt_file(p: Path) -> str:
            try:
                mtime = dt.datetime.fromtimestamp(p.stat().st_mtime, tz=JST)
                return f"{p.name}  （更新: {mtime:%Y-%m-%d %H:%M:%S}）"
            except Exception:
                return p.name

        selected_jsonl: Path = st.selectbox(
            "📘 JSONLログファイルを選択",
            options=jsonl_files,
            index=0,  # name降順なので最新っぽいものが先頭
            format_func=_fmt_file,
        )
    else:
        selected_jsonl = LEGACY_BOT_LOG_FILE  # 存在しない場合もある
        st.warning("ボット用 JSONL が見つかりませんでした。")

    st.divider()
    st.subheader("🔄 ログ更新")

    if st.button("最新ログを読み込み直す", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 選択された JSONL
BOT_LOG_FILE = selected_jsonl.resolve()

# ============================================================
# ログファイル情報（表示）
# ============================================================
with st.expander("📁 ログファイル情報", expanded=True):
    st.write(f"**アプリディレクトリ:** `{APP_DIR}`")
    st.write(f"**ログディレクトリ:** `{LOGS_DIR}`")
    st.write(f"**選択中 JSONL:** `{BOT_LOG_FILE}`")
    st.write(f"**ボットログ(SQLite):** `{SQLITE_DB}`")

    if BOT_LOG_FILE.exists():
        mtime = dt.datetime.fromtimestamp(BOT_LOG_FILE.stat().st_mtime, tz=JST)
        st.write(f"**JSONL 最終更新:** {mtime:%Y-%m-%d %H:%M:%S %Z}")
    else:
        st.warning("選択中の JSONL ログファイルが存在しません。ボット実行後に再度確認してください。")

    if SQLITE_DB.exists():
        st.write("SQLite DB は存在します。")
    else:
        st.info("SQLite DB (`bot_logs.sqlite3`) はまだ存在しません。JSONL だけを利用中の可能性があります。")

# ============================================================
# ログ読み込み（共通lib経由）＋キャッシュ
# ============================================================
@st.cache_data(show_spinner=False)
def get_df_json(path: Path) -> pd.DataFrame:
    return load_jsonl_logs(path)


@st.cache_data(show_spinner=False)
def get_df_sqlite(path: Path) -> pd.DataFrame:
    # log_io 側で table="logs" がデフォルトになっている想定
    return load_sqlite_logs(path)


df_json = get_df_json(BOT_LOG_FILE) if BOT_LOG_FILE.exists() else pd.DataFrame()
df_sql = get_df_sqlite(SQLITE_DB) if SQLITE_DB.exists() else pd.DataFrame()

if df_json.empty and df_sql.empty:
    st.warning("JSONL / SQLite の両方でログデータがまだありません。ボットを利用してから再度開いてください。")
    st.stop()

# ============================================================
# フィルタ UI（期間・アクション・ページ）
# ============================================================
st.divider()
st.subheader("🔍 フィルタ")

# 日付範囲は共通関数で計算
min_date, max_date = calc_date_range(df_json, df_sql)

# action / page の候補も JSONL + SQLite を合わせてユニークに
actions_json = df_json["action"].dropna().unique().tolist() if "action" in df_json.columns else []
actions_sql  = df_sql["action"].dropna().unique().tolist()  if "action" in df_sql.columns  else []
actions = sorted(sorted(set(actions_json) | set(actions_sql)))

pages_json = df_json["page"].dropna().unique().tolist() if "page" in df_json.columns else []
pages_sql  = df_sql["page"].dropna().unique().tolist()  if "page" in df_sql.columns  else []
pages = sorted(sorted(set(pages_json) | set(pages_sql)))

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
        help="例: ask / answer / generate / edit など。未選択なら全種類。",
    )

with c4:
    picked_pages = st.multiselect(
        "ページ（任意）",
        options=pages,
        default=pages,
        help="12_ボット / 22_画像生成 など、同じログファイルを共有している場合に使えます。",
    )

# ============================================================
# JSONL 側フィルタ＆表示（共通フィルタ関数利用）
# ============================================================
if not df_json.empty:
    st.divider()
    st.subheader("📘 JSONL ログ：ユーザー別 全ログ（全項目）")

    fdf_json = apply_common_filters(
        df_json,
        date_from=date_from,
        date_to=date_to,
        actions=picked_actions,
        pages=picked_pages,
    )
    st.caption(f"[JSONL] 対象レコード: **{len(fdf_json):,} / {len(df_json):,}**")

    if not fdf_json.empty:
        sort_cols_json = ["user"]
        if "ts" in fdf_json.columns:
            sort_cols_json.append("ts")
        fdf_json_sorted = fdf_json.sort_values(sort_cols_json, ascending=[True, True])

        st.dataframe(fdf_json_sorted, width="stretch")

        csv_all_json = fdf_json_sorted.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ [JSONL] フィルタ済み全ログ（ユーザー順・全項目）CSV",
            data=csv_all_json,
            file_name="bot_logs_jsonl_all_by_user.csv",
            mime="text/csv",
            width="stretch",
        )

        st.subheader("📜 [JSONL] 個別ユーザーのログ一覧（全項目）")
        target_user_json = st.selectbox(
            "JSONL で詳細を見たいユーザーを選択",
            options=["（未選択）"] + sorted(fdf_json_sorted["user"].unique().tolist()),
            index=0,
            key="user_select_jsonl",
        )
        if target_user_json != "（未選択）":
            udf_json = fdf_json_sorted[fdf_json_sorted["user"] == target_user_json]
            st.caption(f"[JSONL] ユーザー `{target_user_json}` の対象ログ件数: **{len(udf_json):,}**")
            st.dataframe(udf_json, width="stretch")

            csv_user_json = udf_json.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                f"⬇️ [JSONL] ユーザー `{target_user_json}` のログ（全項目）CSV",
                data=csv_user_json,
                file_name=f"bot_logs_jsonl_{target_user_json}_full.csv",
                mime="text/csv",
                width="stretch",
            )
    else:
        st.info("[JSONL] 指定条件に一致するログがありません。")

else:
    st.info("JSONL ログは現在空です（または選択中ファイルが存在しません）。")

# ============================================================
# SQLite 側フィルタ＆表示（共通フィルタ関数利用）
# ============================================================
if not df_sql.empty:
    st.divider()
    st.subheader("🗄️ SQLite ログ：ユーザー別 全ログ（全項目）")

    fdf_sql = apply_common_filters(
        df_sql,
        date_from=date_from,
        date_to=date_to,
        actions=picked_actions,
        pages=picked_pages,
    )
    st.caption(f"[SQLite] 対象レコード: **{len(fdf_sql):,} / {len(df_sql):,}**")

    if not fdf_sql.empty:
        sort_cols_sql = ["user"]
        if "ts" in fdf_sql.columns:
            sort_cols_sql.append("ts")
        fdf_sql_sorted = fdf_sql.sort_values(sort_cols_sql, ascending=[True, True])

        st.dataframe(fdf_sql_sorted, width="stretch")

        csv_all_sql = fdf_sql_sorted.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ [SQLite] フィルタ済み全ログ（ユーザー順・全項目）CSV",
            data=csv_all_sql,
            file_name="bot_logs_sqlite_all_by_user.csv",
            mime="text/csv",
            width="stretch",
        )

        st.subheader("📜 [SQLite] 個別ユーザーのログ一覧（全項目）")
        target_user_sql = st.selectbox(
            "SQLite で詳細を見たいユーザーを選択",
            options=["（未選択）"] + sorted(fdf_sql_sorted["user"].unique().tolist()),
            index=0,
            key="user_select_sqlite",
        )
        if target_user_sql != "（未選択）":
            udf_sql = fdf_sql_sorted[fdf_sql_sorted["user"] == target_user_sql]
            st.caption(f"[SQLite] ユーザー `{target_user_sql}` の対象ログ件数: **{len(udf_sql):,}**")
            st.dataframe(udf_sql, width="stretch")

            csv_user_sql = udf_sql.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                f"⬇️ [SQLite] ユーザー `{target_user_sql}` のログ（全項目）CSV",
                data=csv_user_sql,
                file_name=f"bot_logs_sqlite_{target_user_sql}_full.csv",
                mime="text/csv",
                width="stretch",
            )
    else:
        st.info("[SQLite] 指定条件に一致するログがありません。")

else:
    st.info("SQLite ログは現在空です。`12_ボット.py` 側で SQLite ログを有効にしてからご利用ください。")

# ============================================================
# 終了メッセージ
# ============================================================
st.info(f"✅ 管理者 `{user}` として 40_ログ管理 でボットログ（JSONL / SQLite）を閲覧中です。")
