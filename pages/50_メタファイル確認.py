# pages/50_メタファイル確認.py
# ------------------------------------------------------------
# 🗂️ vectorstore/<vectorspace>/<shard_id>/meta.jsonl ビューア
# - サマリーUIなし
# - シャード選択時のみ遅延読み込み
# - 先に year×pno 集計表 → 次に pno 単一選択で絞り込み
# - 絞り込み結果を Excel でダウンロード可能
# ------------------------------------------------------------

from __future__ import annotations

# --- sys.path 調整（common_lib へ到達） ---
#（pageから）
from pathlib import Path
import sys

_THIS = Path(__file__).resolve()
APP_ROOT = _THIS.parents[1]          # pages -> app root
PROJECTS_ROOT = _THIS.parents[3]     # auth_portal/pages -> projects/auth_portal
import sys
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))


from pathlib import Path
import io
import pandas as pd
import streamlit as st

# 共通ユーティリティ（読み込み用のみ使用）
from lib.rag.vectorstore_utils import iter_jsonl

# パス設定は PATHS に一本化
from config.path_config import PATHS
VS_ROOT: Path = PATHS.vs_root  # => <project>/data/vectorstore

from common_lib.auth.auth_helpers import require_admin_user

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="50 メタファイル確認", page_icon="🗂️", layout="wide")

sub = require_admin_user(st)
if not sub:
    st.error("🚫 このページは管理者のみアクセスできます。")
    st.stop()

st.success(f"✅ 管理者ログイン中: **{sub}**")  # ← 表示はここで自由に

st.title("🗂️ メタファイル確認（シャード選択後に読み込み）")

# ============================================================
# Sidebar: vectorspace 切替
# ============================================================
with st.sidebar:
    st.header("設定")

    # ============================
    # ★ DB切替（vectorspace）
    # ============================
    st.subheader("📚 検索DB（vectorstore）")

    vectorspace_label = st.radio(
        "対象",
        ["report（openai）", "規定集（openai_sample）"],
        index=0,
        help="report: data/vectorstore/openai / 規定集: data/vectorstore/openai_sample",
    )

    VECTORSPACE_MAP = {
        "report（openai）": "openai",
        "規定集（openai_sample）": "openai_sample",
    }
    vectorspace = VECTORSPACE_MAP[vectorspace_label]
    st.caption(f"現在: `{vectorspace}`")

    st.divider()

# ベースディレクトリ確認（vectorspace）
base_backend_dir = VS_ROOT / vectorspace
if not base_backend_dir.exists():
    st.error(f"{base_backend_dir} が見つかりません。先にベクトル化を実行してください。")
    st.stop()

# 利用可能なシャード（= year / current などのフォルダ）を列挙
shard_dirs = sorted([p for p in base_backend_dir.iterdir() if p.is_dir()])
shard_ids = [p.name for p in shard_dirs]

st.subheader("🔍 シャード選択 → 読み込み")
sel_shard = st.selectbox(
    "シャード（year / current）を選択（選択すると読み込み開始）",
    ["（選択してください）"] + shard_ids,
    index=0,
)

if sel_shard == "（選択してください）":
    st.info("上のセレクトボックスから **シャード（year / current）** を選択すると読み込みを開始します。")
    st.stop()

# meta.jsonl
meta_path = base_backend_dir / sel_shard / "meta.jsonl"
if not meta_path.exists():
    st.warning(f"{meta_path} が見つかりません。")
    st.stop()

rows = []
pno_counts: dict[str, int] = {}

with st.spinner(f"シャード {sel_shard} の meta.jsonl を読み込み中…"):
    for obj in iter_jsonl(meta_path):
        o = dict(obj)
        o.setdefault("shard_id", sel_shard)
        o.setdefault("file", o.get("doc_id"))
        rows.append(o)

        p = o.get("pno")
        if p is not None:
            pno_counts[str(p)] = pno_counts.get(str(p), 0) + 1

if not rows:
    st.warning(f"シャード {sel_shard} に表示できるレコードがありません。")
    st.stop()

# DataFrame 化（必要最小列を補完）
df = pd.DataFrame(rows)
for col in [
    "file", "page", "chunk_id", "chunk_index", "text",
    "span_start", "span_end", "shard_id", "year", "pno"
]:
    if col not in df.columns:
        df[col] = None

df["chunk_len"] = df["text"].astype(str).str.len()

# ============================================================
# 📅 year × pno 組み合わせ（出現件数）
# ============================================================
yp_df = (
    df.assign(year=pd.to_numeric(df["year"], errors="coerce"))
      .dropna(subset=["year", "pno"])
)
if not yp_df.empty:
    yp_df["year"] = yp_df["year"].astype(int)
    year_pno_df = (
        yp_df.groupby(["year", "pno"])
             .size()
             .reset_index(name="rows")
             .sort_values(["year", "pno"], kind="mergesort")
    )
else:
    year_pno_df = pd.DataFrame(columns=["year", "pno", "rows"])

st.subheader("📅 year × pno の組み合わせ（出現件数）")
st.caption(f"DB: `{vectorspace}` / shard: `{sel_shard}`")
if year_pno_df.empty:
    st.info("このシャード内に year / pno の組み合わせはありません。")
else:
    st.dataframe(year_pno_df, width='stretch', height=360)

# ============================================================
# 🎯 pno 単一選択 → 絞り込み表示
# ============================================================
pno_options = sorted(pno_counts.keys())

sel_pno = st.selectbox(
    "pno を選択（このシャード内のみ）",
    ["（選択してください）"] + pno_options,
    index=0,
    help="選んだシャード内の pno だけが候補になります。",
)

if sel_pno == "（選択してください）":
    st.info("pno を選択すると該当レコードが表示されます。")
    st.stop()

filtered_df = df[df["pno"].astype(str) == sel_pno]

st.caption(
    f"📊 該当レコード数: {len(filtered_df):,} 件（pno={sel_pno}）｜"
    f"DB: {vectorspace}｜シャード: {sel_shard}"
)

st.dataframe(filtered_df, width='stretch', height=600)

# ============================================================
# 📥 ダウンロード（絞り込み済みのみ）
# ============================================================
xlsx_bytes = io.BytesIO()
with pd.ExcelWriter(xlsx_bytes, engine="xlsxwriter") as writer:
    filtered_df.to_excel(writer, index=False, sheet_name="rows")

st.download_button(
    label="📥 この絞り込み結果をExcel（.xlsx）でダウンロード",
    data=xlsx_bytes.getvalue(),
    file_name=f"meta_rows_db_{vectorspace}_shard_{sel_shard}_pno_{sel_pno}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
