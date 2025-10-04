# pages/05_pdfベクトル化.py
# ------------------------------------------------------------
# 📥 <PDF_ROOT>/<shard>/<pno> を取り込み、
#    <VS_ROOT>/<backend>/<shard>/ に vectors.npy / meta.jsonl を追記。
#    meta には year / pno / page / embed_model / shard_id / chunk_len_tokens / ocr 等を付与。
#    重要: meta.jsonl への追記は NumpyVectorDB.add() が行うため二重追記しない。
#    ※ OpenAI の埋め込みモデルは text-embedding-3-large に固定（3072 次元）
# ------------------------------------------------------------

"""PDFベクトル化ユーティリティ（page 単位 with year/pno メタ, _ocr 優先）

本モジュールは、指定された PDF ルート（PDF_ROOT）配下の「シャード（shard=年度フォルダ）」を単数選択し、
その直下の「プロジェクト番号フォルダ（pno）」を選択して、当該フォルダ内の PDF を取り込みます。

・_ocr.pdf を優先採用（例: A.pdf と A_ocr.pdf が同居 → A_ocr.pdf のみ取り込み、A.pdf はスキップ）
・meta へ ocr 状態（"done"/"no"）、year、pno を付与
・NumpyVectorDB（<VS_ROOT>/<backend>/<shard>/）に vectors.npy / meta.jsonl を追記
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import json

import streamlit as st
import pdfplumber
import numpy as np
import tiktoken

from config.path_config import PATHS
from config import pricing
from lib.rag_utils import split_text, EmbeddingStore, NumpyVectorDB, ProcessedFilesSimple
from lib.vectorstore_utils import load_processed_files, save_processed_files
from lib.text_normalize import normalize_ja_text

# ============================================================
# 定数（Constants）
# ============================================================
OPENAI_EMBED_MODEL = "text-embedding-3-large"  # 固定（3072 次元）

# ============================================================
# tokenizer 準備（Tokenizer setup）
# ============================================================
try:
    enc = tiktoken.encoding_for_model(OPENAI_EMBED_MODEL)
except Exception:
    enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """テキストのトークン数（token count）を返す。"""
    return len(enc.encode(text))

# ============================================================
# UI（Streamlit UI）
# ============================================================
st.set_page_config(page_title="05 ベクトル化（page・year/pno対応・_ocr優先）", page_icon="🧱", layout="wide")
st.title("🧱 年度（=シャード）→ プロジェクト番号（=フォルダ）ごとのベクトル化")

with st.sidebar:
    st.subheader("📓 現在のロケーション（表示のみ）")
    st.markdown(f"**Location:** `{PATHS.preset}`")
    st.markdown("#### 📂 解決パス（コピー可）")
    st.text_input("PDF_ROOT", str(PATHS.pdf_root), key="p_pdf", disabled=True)
    if hasattr(PATHS, "data_root"):
        st.text_input("DATA_ROOT", str(PATHS.data_root), key="p_data", disabled=True)
    st.text_input("VS_ROOT",  str(PATHS.vs_root),  key="p_vs",  disabled=True)

PDF_ROOT: Path = PATHS.pdf_root
VS_ROOT: Path  = PATHS.vs_root

# パラメータ
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    backend = st.radio("埋め込みバックエンド", ["openai", "local"], index=0, horizontal=True)
    if backend == "openai":
        st.caption(f"🔧 Embedding モデルは **{OPENAI_EMBED_MODEL}（3072次元）固定**")
with col2:
    chunk_size = st.number_input("チャンクサイズ（文字）", 200, 3000, 900, 50)
    overlap    = st.number_input("オーバーラップ（文字）", 0, 600, 150, 10)
with col3:
    batch_size = st.number_input("埋め込みバッチ数", 8, 512, 64, 8)
    st.caption("※ OCR が必要な PDF は事前に検索可能 PDF 化（ocrmypdf 等）推奨。")

# 入出力の実パス
st.info(
    f"**入力**: `{PDF_ROOT}/<shard>/<pno>`\n\n"
    f"**出力**: `{VS_ROOT}/{backend}/<shard>`"
)

# ============================================================
# パス・列挙ヘルパ（Path helpers）
# ============================================================
def list_shards() -> List[str]:
    """PDF_ROOT 直下の年度フォルダ（シャード）一覧。"""
    if not PDF_ROOT.exists():
        return []
    return sorted([p.name for p in PDF_ROOT.iterdir() if p.is_dir()])

def list_pnos(shard_id: str) -> List[str]:
    """指定 shard 配下のプロジェクト番号フォルダ（pno）一覧。"""
    d = PDF_ROOT / shard_id
    if not d.exists():
        return []
    return sorted([p.name for p in d.iterdir() if p.is_dir()])

def list_pdfs_in_pno(shard_id: str, pno: str) -> List[Path]:
    """指定 shard/pno 配下の *.pdf 一覧。"""
    d = PDF_ROOT / shard_id / pno
    if not d.exists():
        return []
    return sorted(d.glob("*.pdf"))

def ensure_vs_dir(backend: str, shard_id: str) -> Path:
    """<VS_ROOT>/<backend>/<shard> を作成（なければ）。"""
    d = VS_ROOT / backend / shard_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_vector_count(base_dir: Path) -> int:
    """vectors.npy の行数（総ベクトル数）を返す。"""
    p = base_dir / "vectors.npy"
    if not p.exists():
        return 0
    try:
        arr = np.load(p, mmap_mode="r")
        return int(arr.shape[0]) if arr.ndim == 2 else 0
    except Exception:
        return 0

# ============================================================
# processed_files.json の canon 化（shard/pno/filename 形式）
# ============================================================
def migrate_processed_files_to_canonical(pf_json: Path, shard_id: str, pno: str) -> None:
    """旧フォーマット（'filename' または 'shard/filename'）を 'shard/pno/filename' へ正規化。"""
    pf_list = load_processed_files(pf_json)
    if not pf_list:
        return
    changed = False
    canonical: List[str] = []
    for entry in pf_list:
        if isinstance(entry, str):
            val = entry
        elif isinstance(entry, dict):
            val = entry.get("file") or entry.get("path") or entry.get("name")
        else:
            continue
        if not val:
            continue
        parts = val.split("/")
        if len(parts) == 1:
            # 'filename' → 'shard/pno/filename'
            val = f"{shard_id}/{pno}/{val}"
            changed = True
        elif len(parts) == 2:
            # 'shard/filename' → 'shard/pno/filename'
            if parts[0] == shard_id:
                val = f"{shard_id}/{pno}/{parts[1]}"
                changed = True
        # len(parts) >= 3 はそのまま採用
        canonical.append(val)

    # 重複除去（順序維持）
    seen, dedup = set(), []
    for v in canonical:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)

    if changed:
        save_processed_files(pf_json, dedup)

# ============================================================
# _ocr 優先の取り込み候補決定
# ============================================================
def decide_ocr_candidates(pdf_paths: List[Path]) -> Tuple[List[Tuple[Path, str]], List[str]]:
    """同一ベース名の A.pdf / A_ocr.pdf がある場合は _ocr を採用し、A.pdf をスキップする。

    Returns:
        candidates: [(採用する Path, ocr_flag("done" or "no")), ...]
        logs:      [ユーザー表示用の判断ログ文字列]
    """
    by_base: Dict[str, Dict[str, Path]] = {}
    for p in pdf_paths:
        stem = p.stem
        if stem.endswith("_ocr"):
            base = stem[:-4]  # remove '_ocr'
            by_base.setdefault(base, {})["ocr"] = p
        else:
            base = stem
            by_base.setdefault(base, {})["base"] = p

    candidates: List[Tuple[Path, str]] = []
    logs: List[str] = []

    for base, d in by_base.items():
        ocr = d.get("ocr")
        basep = d.get("base")
        if ocr is not None:
            candidates.append((ocr, "done"))
            if basep is not None:
                logs.append(f"🟢 採用: {ocr.name}（_ocr優先） / ⏭️ スキップ: {basep.name}")
            else:
                logs.append(f"🟢 採用: {ocr.name}（_ocr 単独）")
        elif basep is not None:
            candidates.append((basep, "no"))
            logs.append(f"🟢 採用: {basep.name}（_ocr なし）")
        # どちらも無いケースは実質起きない

    # ファイル名昇順に安定化
    candidates.sort(key=lambda t: t[0].name)
    return candidates, logs

# ============================================================
# シャード→pno 選択 UI
# ============================================================
shards = list_shards()
if not shards:
    st.warning(f"{PDF_ROOT} 配下に年度フォルダー（=シャード）がありません。例: {PDF_ROOT}/2025/<pno>/*.pdf")
    st.stop()

selected_shard = st.selectbox("対象シャード（年度）を選択", shards, index=0)
pnos = list_pnos(selected_shard)
if not pnos:
    st.warning(f"{PDF_ROOT}/{selected_shard} にプロジェクト番号フォルダがありません。")
    st.stop()

selected_pnos = st.multiselect("対象プロジェクト番号（pno）を選択（複数可）", pnos, default=pnos[:1])

st.info(
    "選択した **年度/プロジェクト** 配下の PDF を取り込みます。"
    " `_ocr.pdf` がある場合はそれを採用し、同名の元 PDF はスキップします。"
)

run = st.button("選択した pno フォルダ内の PDF を取り込み", type="primary")

# ============================================================
# 実行
# ============================================================
if run:
    estore = EmbeddingStore(backend=backend, openai_model=OPENAI_EMBED_MODEL)
    total_files = 0
    total_chunks = 0

    overall_progress = st.progress(0.0, text="準備中…")
    file_progress = st.progress(0.0, text="ファイル進捗：待機中…")
    status_current = st.empty()

    num_pnos = len(selected_pnos)
    # 年度情報
    try:
        year_val = int(selected_shard)
    except ValueError:
        year_val = None

    # ベクトル出力（シャード単位で保存）
    vs_dir = ensure_vs_dir(backend, selected_shard)
    tracker = ProcessedFilesSimple(vs_dir / "processed_files.json")
    vdb = NumpyVectorDB(vs_dir)

    # ---- pno ごとのループ ----
    for i_pno, pno in enumerate(selected_pnos, start=1):
        st.markdown(f"### 📁 プロジェクト: `{selected_shard}/{pno}`")

        # 旧 processed_files を 'shard/pno/filename' に正規化
        migrate_processed_files_to_canonical(vs_dir / "processed_files.json", selected_shard, pno)

        # 候補一覧（_ocr 優先の採用決定＋ログ）
        raw_pdfs = list_pdfs_in_pno(selected_shard, pno)
        candidates, ocr_logs = decide_ocr_candidates(raw_pdfs)

        with st.expander("🧾 取り込み前の判定ログ（_ocr 優先の採否）", expanded=True):
            for line in ocr_logs:
                st.write(line)
            if not candidates:
                st.info("この pno には取り込み対象の PDF がありません。")

        if not candidates:
            overall_progress.progress(i_pno / num_pnos, text=f"{i_pno}/{num_pnos} pno 完了（空）")
            continue

        pno_new_files = 0
        pno_new_chunks = 0

        # ---- 採用候補ごとの処理 ----
        for i_file, (pdf_path, ocr_flag) in enumerate(candidates, start=1):
            name = pdf_path.name
            key_full = f"{selected_shard}/{pno}/{name}"  # 正準キー

            # 旧キー（shard/name, name）もスキップ対象に含める（後方互換）
            if tracker.is_done(key_full) or tracker.is_done(f"{selected_shard}/{name}") or tracker.is_done(name):
                status_current.info(f"⏭️ スキップ（既取込）: `{selected_shard}/{pno}` / **{name}**")
                file_progress.progress(1.0, text=f"ファイル {i_file}/{len(candidates)} 完了: {name}")
                continue

            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    total_pages = max(len(pdf.pages), 1)
                    status_current.info(
                        f"📥 取り込み開始: `{selected_shard}/{pno}` / **{name}**（{i_file}/{len(candidates)}） 全{total_pages}ページ"
                    )
                    file_progress.progress(0.0, text=f"ファイル {i_file}/{len(candidates)}: {name} - 0/{total_pages} ページ")

                    # ---- ページごとに抽出→分割→埋め込み ----
                    for page_no, page in enumerate(pdf.pages, start=1):
                        raw = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
                        raw = raw.replace("\t", " ").replace("\xa0", " ")
                        text = " ".join(raw.split())

                        if not text:
                            file_progress.progress(page_no / total_pages,
                                text=f"ファイル {i_file}/{len(candidates)}: {name} - {page_no}/{total_pages} ページ")
                            continue

                        text = normalize_ja_text(text)
                        spans: List[Tuple[str, int, int]] = split_text(
                            text, chunk_size=int(chunk_size), overlap=int(overlap)
                        )
                        if not spans:
                            file_progress.progress(page_no / total_pages,
                                text=f"ファイル {i_file}/{len(candidates)}: {name} - {page_no}/{total_pages} ページ")
                            continue

                        chunks = [s[0] for s in spans]
                        vectors: List[np.ndarray] = []
                        metas: List[dict] = []

                        for i in range(0, len(chunks), int(batch_size)):
                            batch = chunks[i:i + int(batch_size)]
                            vecs = estore.embed(batch, batch_size=int(batch_size)).astype("float32")
                            vectors.append(vecs)

                            for j, (ch, s, e) in enumerate(spans[i:i + int(batch_size)]):
                                metas.append({
                                    "file": key_full,                           # 'shard/pno/filename'
                                    "year": year_val,                           # 例: 2025
                                    "pno": pno,                                 # プロジェクト番号
                                    "page": page_no,
                                    "chunk_id": f"{name}#p{page_no}-{i+j}",
                                    "chunk_index": i + j,
                                    "text": ch,
                                    "span_start": s,
                                    "span_end": e,
                                    "chunk_len_tokens": count_tokens(ch),
                                    "ingested_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                    "shard_id": selected_shard,
                                    "embed_model": OPENAI_EMBED_MODEL if backend == "openai" else "local-model",
                                    "ocr": "done" if ocr_flag == "done" else "no",
                                })

                        vec_mat = np.vstack(vectors) if len(vectors) > 1 else vectors[0]
                        vdb.add(vec_mat, metas)
                        pno_new_chunks += len(metas)

                        file_progress.progress(page_no / total_pages,
                            text=f"ファイル {i_file}/{len(candidates)}: {name} - {page_no}/{total_pages} ページ")

                tracker.mark_done(key_full)
                pno_new_files += 1
                status_current.success(f"✅ 完了: `{selected_shard}/{pno}` / **{name}**（{i_file}/{len(candidates)}）")

            except Exception as e:
                st.error(f"❌ 取り込み失敗: {name} : {e}")
                status_current.error(f"❌ 失敗: `{selected_shard}/{pno}` / **{name}** - {e}")

            overall_progress.progress(
                (i_pno - 1 + i_file / max(len(candidates), 1)) / num_pnos,
                text=f"全体 {i_pno}/{num_pnos} pno 処理中…（{pno}: {i_file}/{len(candidates)} ファイル）"
            )

        st.success(f"📁 `{selected_shard}/{pno}`: 新規ファイル {pno_new_files} 件 / 追加チャンク {pno_new_chunks} 件")
        st.caption(f"🔎 シャード内ベクトル総数（DB計測）: {get_vector_count(vs_dir):,d}")

        overall_progress.progress(i_pno / num_pnos, text=f"{i_pno}/{num_pnos} pno 完了")

        total_files  += pno_new_files
        total_chunks += pno_new_chunks

    st.toast(f"✅ 完了: 新規 {total_files} ファイル / {total_chunks} チャンク（_ocr優先・year/pno付き）", icon="✅")

    # ---------- 料金計算（openai backend のみ） ----------
    if total_chunks > 0:
        st.markdown("### 💰 埋め込みコストの概算")
        if backend == "openai":
            total_tokens = 0
            meta_path = (VS_ROOT / backend / selected_shard / "meta.jsonl")
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        total_tokens += int(obj.get("chunk_len_tokens", 0))
            model = OPENAI_EMBED_MODEL
            usd = pricing.estimate_embedding_cost_usd(total_tokens, model)
            jpy = pricing.estimate_embedding_cost_jpy(total_tokens, model)
            st.write(f"- モデル: **{model}**")
            st.write(f"- 総トークン数: {total_tokens:,}")
            st.write(f"- 概算コスト: `${usd:.4f}` ≈ ¥{jpy:,.0f}")
        else:
            st.info("local backend のためコストは発生しません。")
