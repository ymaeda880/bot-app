# pages/50_メタファイル確認.py
# ------------------------------------------------------------
# 🗂️ vectorstore/(openai|local)/<shard_id>/meta.jsonl ビューア
# - 複数シャードの meta.jsonl を横断して読み込み・絞り込み・CSV/Excel 出力
# - 🚫 メンテナンス機能（削除・バックアップ等）は含めない
# - ★ タイトル直下に「対象シャードの year / pno 一覧」＋ Excel ダウンロードを表示
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import io
import pandas as pd
import streamlit as st

# 共通ユーティリティ（読み込み用のみ使用）
from lib.rag.vectorstore_utils import iter_jsonl

# パス設定は PATHS に一本化
from config.path_config import PATHS
VS_ROOT: Path = PATHS.vs_root  # => <project>/data/vectorstore


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="50 メタファイル確認", page_icon="🗂️", layout="wide")
st.title("🗂️ メタファイル確認（フォルダー＝シャード）")

# ★ タイトル直下のプレースホルダ（year/pno サマリーを後でここへ描画）
summary_box = st.container()

# --- サイドバー ---
with st.sidebar:
    st.header("読み込み設定")
    backend = st.radio("バックエンド", ["openai", "local"], index=0, horizontal=True)

    base_backend_dir = VS_ROOT / backend
    if not base_backend_dir.exists():
        st.error(f"{base_backend_dir} が見つかりません。先にベクトル化を実行してください。")
        st.stop()

    shard_dirs = sorted([p for p in base_backend_dir.iterdir() if p.is_dir()])
    shard_ids = [p.name for p in shard_dirs]
    sel_shards = st.multiselect("対象シャード（複数可）", shard_ids, default=shard_ids)
    max_rows_read = st.number_input("各シャードの最大行数（0=全件）", min_value=0, value=0, step=1000)

if not sel_shards:
    st.warning("少なくとも1つのシャードを選択してください。")
    st.stop()

# --- データ読み込み ---
all_rows = []
for sid in sel_shards:
    meta_path = base_backend_dir / sid / "meta.jsonl"
    read_cnt = 0
    for obj in iter_jsonl(meta_path):
        obj = dict(obj)
        obj.setdefault("shard_id", sid)
        obj.setdefault("file", obj.get("doc_id"))
        all_rows.append(obj)
        read_cnt += 1
        if max_rows_read and read_cnt >= max_rows_read:
            break

if not all_rows:
    st.warning("表示できるレコードがありません。")
    st.stop()

df = pd.DataFrame(all_rows)

# 足りない列を補完（存在しない場合は None）
for col in ["file","page","chunk_id","chunk_index","text","span_start","span_end","shard_id","year","pno"]:
    if col not in df.columns:
        df[col] = None

df["chunk_len"] = df["text"].astype(str).str.len()

# ============================================================
# ★ タイトル直下に「year / pno の一覧と Excel ダウンロード」を表示
# ============================================================
with summary_box:
    st.subheader("🧭 サマリー（year / pno 一覧）")

    # year 一覧（値の正規化：数値化できるものは int 化）
    years_series = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    years_df = (
        years_series.value_counts()
        .rename("rows")
        .to_frame()
        .sort_index()
        .rename_axis("year")
        .reset_index()
    )

    # pno 一覧（文字列化して集計）
    pnos_series = df["pno"].dropna().astype(str)
    pnos_df = (
        pnos_series.value_counts()
        .rename("rows")
        .to_frame()
        .sort_index()
        .rename_axis("pno")
        .reset_index()
    )

    # year × pno 組み合わせ
    yp_df = (
        df.assign(year=pd.to_numeric(df["year"], errors="coerce"))
          .dropna(subset=["year","pno"])
    )
    if not yp_df.empty:
        yp_df["year"] = yp_df["year"].astype(int)
        year_pno_df = (
            yp_df.groupby(["year","pno"])
                .size()
                .reset_index(name="rows")
                .sort_values(["year","pno"], kind="mergesort")
        )
    else:
        year_pno_df = pd.DataFrame(columns=["year","pno","rows"])

    # 表示（小さめの先頭）
    c1, c2 = st.columns(2)
    with c1:
        st.caption("存在する year（出現件数）")
        st.dataframe(years_df.head(200), use_container_width=True, height=260)
    with c2:
        st.caption("存在する pno（出現件数）")
        st.dataframe(pnos_df.head(200), use_container_width=True, height=260)

    # ============================================================
    # year × pno 組み合わせ（出現件数）を年ごとに選択表示（常時表示版）
    # ============================================================
    st.subheader("📅 year × pno の組み合わせ（出現件数）")

    if year_pno_df.empty:
        st.info("データがありません。")
    else:
        # 利用可能な year 一覧（ソート）
        year_opts = sorted(year_pno_df["year"].unique().tolist())
        sel_year_for_combo = st.selectbox("表示する year を選択", year_opts)

        # 該当 year のみ抽出
        filtered_combo = year_pno_df[year_pno_df["year"] == sel_year_for_combo]

        st.caption(f"year = {sel_year_for_combo} に含まれる pno 一覧（件数: {len(filtered_combo):,}）")
        st.dataframe(filtered_combo, use_container_width=True, height=360)



    # Excel（.xlsx）作成（複数シート）
    xlsx_bytes = io.BytesIO()
    with pd.ExcelWriter(xlsx_bytes, engine="xlsxwriter") as writer:
        # 選択シャード一覧
        pd.DataFrame({"shard_id": sel_shards}).to_excel(writer, index=False, sheet_name="shards")
        # 各サマリー
        years_df.to_excel(writer, index=False, sheet_name="years")
        pnos_df.to_excel(writer, index=False, sheet_name="pno")
        year_pno_df.to_excel(writer, index=False, sheet_name="year_pno")
        # 元データの最小限（必要なら全列に変更）
        cols_export = ["shard_id","file","page","year","pno","chunk_id","chunk_index","span_start","span_end","chunk_len"]
        export_df = df[[c for c in cols_export if c in df.columns]].copy()
        export_df.to_excel(writer, index=False, sheet_name="sample_rows")

    st.download_button(
        label="📥 サマリーをExcel（.xlsx）でダウンロード",
        data=xlsx_bytes.getvalue(),
        file_name="meta_summary_year_pno.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.divider()

# ============================================================
# 📋 year / pno 絞り込み表示（単一選択版）
# ============================================================
st.subheader("🔍 year / pno で絞り込み表示")

# 利用可能な year, pno の一覧を取得（NaN 除去・ソート）
year_options = sorted(df["year"].dropna().astype(str).unique().tolist())
pno_options  = sorted(df["pno"].dropna().astype(str).unique().tolist())

col1, col2 = st.columns([1, 1])
with col1:
    sel_year = st.selectbox("year を選択", ["（すべて）"] + year_options, index=0)
with col2:
    sel_pno = st.selectbox("pno を選択", ["（すべて）"] + pno_options, index=0)

# year/pno でフィルタ
filtered_df = df.copy()
if sel_year != "（すべて）":
    filtered_df = filtered_df[filtered_df["year"].astype(str) == sel_year]
if sel_pno != "（すべて）":
    filtered_df = filtered_df[filtered_df["pno"].astype(str) == sel_pno]

st.caption(f"📊 該当レコード数: {len(filtered_df):,} 件")

# 結果を表示
if filtered_df.empty:
    st.warning("該当するレコードがありません。")
else:
    st.dataframe(filtered_df, use_container_width=True, height=600)

st.divider()
