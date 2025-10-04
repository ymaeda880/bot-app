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
import json
import pandas as pd
import streamlit as st

# 共通ユーティリティ（読み込み用のみ使用）
from lib.vectorstore_utils import iter_jsonl

# パス設定は PATHS に一本化
from config.path_config import PATHS
VS_ROOT: Path = PATHS.vs_root  # => <project>/data/vectorstore

# ============================================================
# クリップボードコピー（JS埋め込み）
# ============================================================
def copy_button(text: str, label: str, key: str):
    payload = json.dumps(text, ensure_ascii=False)
    html = f"""
    <button id="{key}" style="
        padding:6px 10px;border-radius:8px;border:1px solid #dadce0;
        background:#fff;cursor:pointer;font-size:0.9rem;">📋 {label}</button>
    <script>
      const btn = document.getElementById("{key}");
      if (btn) {{
        btn.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText({payload});
            const old = btn.innerText;
            btn.innerText = "✅ コピーしました";
            setTimeout(()=>{{ btn.innerText = old; }}, 1200);
          }} catch(e) {{
            console.error(e);
            alert("コピーに失敗しました: " + e);
          }}
        }});
      }}
    </script>
    """
    st.components.v1.html(html, height=38)

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

    with st.expander("year × pno の組み合わせ（出現件数）", expanded=False):
        st.dataframe(year_pno_df.head(500), use_container_width=True, height=360)

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

# 先頭表示（既存）
st.dataframe(df.head(500), width="stretch", height=560)
st.divider()

# ============================================================
# 📋 ファイル名コピー
# ============================================================
st.subheader("📋 ファイル名コピー（year/file.pdf）")
file_list = sorted(df["file"].dropna().unique().tolist())
q = st.text_input("ファイル名フィルタ", value="")
filtered = [f for f in file_list if q.lower() in str(f).lower()] if q else file_list
st.caption(f"ヒット: {len(filtered)} 件")
cols = st.columns(3)
for i, f in enumerate(filtered[:100]):
    with cols[i % 3]:
        st.write(f"`{f}`")
        copy_button(text=f, label="コピー", key=f"copy_{i}")
