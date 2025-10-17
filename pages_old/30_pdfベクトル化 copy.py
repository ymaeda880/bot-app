
# pages/30_pdfベクトル化.py

"""
# ─────────────────────────────────────────────────────────────
# File: pages/30_pdfベクトル化.py
# 概要: UI と最小限の制御だけを担当（ロジックは lib/pdf_ingest.py に集約）
# ─────────────────────────────────────────────────────────────

05_pdfベクトル化 — Streamlit UI（page・year/pno 対応、_ocr 優先）

目的
----
PDF ベクトル化パイプラインの **UI 層と最小の制御ロジック** を担うページです。
実際の業務ロジック（シャード／pno 列挙、OCR 優先の採否決定、processed_files 正規化、
ページテキスト抽出→チャンク化→埋め込み→ベクトルDB追記 など）は、
すべて `lib/pdf_ingest.py` に切り出されています。

特徴
----
- 初期化～処理中の「見えない待ち時間」を可視化するため、段階的な進捗表示を強化
  - フェーズ表示（準備→ベクトルストア準備→pno処理→完了）
  - ライブ統計（新規/既取込/side除外/_skip除外/失敗/追加チャンク）
  - ETA（単純移動平均ベース）
  - 各ファイル処理のミニ進捗を表示

想定ディレクトリ構成と入出力
----------------------------
- 入力：`PDF_ROOT/<shard>/<pno>/*.pdf`（.pdf / .PDF の双方を対象、再帰なし）
- 出力：`VS_ROOT/<backend>/<shard>/vectors.npy` / `VS_ROOT/<backend>/<shard>/meta.jsonl`
- 処理済み管理：`VS_ROOT/<backend>/<shard>/processed_files.json`（ProcessedFilesSimple）

除外ルール
-----------
- ファイル名に `_skip` を含む PDF は除外（例: `xxx_skip.pdf`, `yyy_ocr_skip.pdf`）
- `<basename>_side.json` の `"ocr"` が以下のいずれかの場合は除外  
  → `"skipped"`, `"failed"`, `"locked"`
- `_ocr.pdf` と素の `*.pdf` が同一ベース名で同居している場合、**_ocr のみ採用**
- ページ単位の抽出で **空文字（画像のみページなど）** は lib 側で自動スキップ
- 例外発生時も、UI はエラーを表示して他ファイル処理を継続

UIの主な役割
-------------
1. 現在のルートパス（PATHS）をサイドバーで表示（確認用）
2. 埋め込みバックエンド・chunking・batching 等のパラメータを設定
3. 対象シャード（年度）・pno を選択
4. `_ocr` 優先の採否ルールを反映した「未処理プレビュー」を提示
5. 実行ボタンで ingest（取り込み）を開始し、進捗・サマリ・概算コストを表示
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import time

import streamlit as st

from config.path_config import PATHS
from lib.costs import estimate_embedding_cost, summarize_embedding_cost_from_meta, DEFAULT_USDJPY
from lib.rag.rag_utils import EmbeddingStore, NumpyVectorDB, ProcessedFilesSimple

# 主要ロジックは lib 側へ分離（UI から関数を呼び出すだけ）
from lib.pdf_ingest import (
    OPENAI_EMBED_MODEL,
    list_shards, list_pnos, list_pdfs_in_pno, ensure_vs_dir,
    compute_unprocessed_map, decide_ocr_candidates,
    migrate_processed_files_to_canonical, ingest_pdf_file,
    get_vector_count,
)



# ============================================================
# ユーティリティ（_skip / _side.json(ocr in {skipped,failed,locked}) 除外）
# ============================================================
def _is_skip_file(p: Path) -> bool:
    """ベース名に '_skip' を含む PDF を除外対象にする（大文字・小文字を区別しない）。"""
    return "_skip" in p.stem.lower()

_EXCLUDED_SIDE_OCR = {"skipped", "failed", "locked"}

def _is_side_excluded(p: Path) -> Tuple[bool, str]:
    """
    <basename>_side.json に ocr が {skipped, failed, locked} のいずれかなら除外。
    返り値: (is_excluded, reason)
    """
    side_path = p.with_name(p.stem + "_side.json")
    if not side_path.exists():
        return False, ""
    try:
        meta = json.loads(side_path.read_text(encoding="utf-8"))
    except Exception:
        # 読めない場合は除外しない（ログ過多回避）
        return False, ""
    ocr_val = str(meta.get("ocr", "")).lower()
    if ocr_val in _EXCLUDED_SIDE_OCR:
        return True, f"side.json の ocr:'{ocr_val}'"
    return False, ""

def _filter_skip(paths: List[Path]) -> Tuple[List[Path], List[str]]:
    """
    `_skip` を含むファイル名、または `<basename>_side.json` の ocr が
    {skipped, failed, locked} の PDF を除外。ログはユーザー確認用に返す。
    """
    kept: List[Path] = []
    logs: List[str] = []
    for p in paths:
        # ① ファイル名 _skip
        if _is_skip_file(p):
            logs.append(f"不採用: {p.name} — `_skip` のため除外")
            continue
        # ② side.json による除外
        is_ex, reason = _is_side_excluded(p)
        if is_ex:
            logs.append(f"不採用: {p.name} — {reason}")
            continue
        kept.append(p)
    return kept, logs

# ------------------------------------------------------------
# 表示補助
# ------------------------------------------------------------
def _fmt_eta(sec: float) -> str:
    if sec <= 0 or sec != sec:
        return "—"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}時間{m:02d}分{s:02d}秒"
    if m > 0:
        return f"{m:d}分{s:02d}秒"
    return f"{s:d}秒"

# ============================================================
# ページ基本設定（タイトルやレイアウト）
# ============================================================
st.set_page_config(
    page_title="05 ベクトル化（page・year/pno対応・_ocr優先）",
    page_icon="🧱",
    layout="wide",
)
st.title("🧱 年度（=シャード）→ プロジェクト番号（=フォルダ）ごとのベクトル化")

st.markdown("""
**処理の流れ:**
1. doc-manger-appの「PDFビューア（ocr）」で画像pdfをテキストpdfに変えてからこのベクトル化処理を行う  
2. テキスト化されたpdfには `_ocr` がファイル名につくので、ここでは `_ocr` ファイルが処理され、元となった pdf は除外される  
3. `_ocr` となっても、判定では画像 pdf となるページが含まれる場合があるが、**存在するテキストのみ** でベクトル化される  
4. **`*_skip*.pdf`** または **`<basename>_side.json` に `ocr:"skipped" / "failed" / "locked"`** があるものは一律で不採用（除外）  
""")

# ------------------------------------------------------------
# サイドバー：現在のパス状況を表示（操作は不可）
# ------------------------------------------------------------
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

# ============================================================
# 取り込みパラメータ（backend / chunking / batching）
# ============================================================
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

st.info(f"**入力**: `{PDF_ROOT}/<shard>/<pno>`\n\n**出力**: `{VS_ROOT}/{backend}/<shard>`")

# ============================================================
# シャード / pno 選択
# ============================================================
shards = list_shards(PDF_ROOT)
if not shards:
    st.warning(f"{PDF_ROOT} 配下に年度フォルダー（=シャード）がありません。例: {PDF_ROOT}/2025/<pno>/*.pdf")
    st.stop()

selected_shard = st.selectbox("対象シャード（年度）を選択", shards, index=0)

pnos = list_pnos(PDF_ROOT, selected_shard)
if not pnos:
    st.warning(f"{PDF_ROOT}/{selected_shard} にプロジェクト番号フォルダがありません。")
    st.stop()

# ============================================================
# 未処理プレビュー（チェック → 下の multiselect へ同期）
# ============================================================
st.markdown("#### 🔎 このシャードの処理状況（pnoごとプレビュー）")
status_map = compute_unprocessed_map(PDF_ROOT, VS_ROOT, selected_shard, backend)

checked_pnos: List[str] = []
for pno in sorted(status_map.keys()):
    info = status_map[pno]
    status = info["status"]; total = info["total_candidates"]

    if status == "none":
        label = f"⚪ **{pno}** — 完全未処理（対象 {total} 件）"
    elif status == "complete":
        label = f"✅ **{pno}** — 処理済（対象 {total} 件）"
    else:
        unproc = info.get("unprocessed_files", []) or []
        label = f"🟡 **{pno}** — 一部未処理（残り {len(unproc)} / {total} 件）"

    key = f"pno_check::{selected_shard}::{pno}"
    if st.checkbox(label, key=key, value=st.session_state.get(key, False)):
        checked_pnos.append(pno)

    if status == "partial":
        unproc = info.get("unprocessed_files", []) or []
        if unproc:
            with st.expander(f"未処理ファイル（{len(unproc)}件） — {pno}", expanded=False):
                def _ellipsis(s: str, n: int = 40) -> str:
                    return s if len(s) <= n else s[:n-1] + "…"
                st.code("\n".join(_ellipsis(x) for x in unproc), language="text")

if checked_pnos:
    st.session_state["selected_pnos"] = checked_pnos

selected_pnos = st.multiselect(
    "対象プロジェクト番号（pno）を選択（複数可）",
    pnos,
    default=st.session_state.get("selected_pnos", (checked_pnos if checked_pnos else pnos[:1])),
    key="selected_pnos",
)

st.info("選択した **年度/プロジェクト** 配下の PDF を取り込みます。 `_ocr.pdf` がある場合はそれを採用します。`*_skip*.pdf` と `side.json(ocr:\"skipped/failed/locked\")` は除外します。")

# ============================================================
# 🧾 取り込み前の判定ログ（ボタン押下で表示）
# ============================================================
show_logs = st.checkbox("🧾 取り込み前の判定ログ（_ocr 優先・_skip・side.json: skipped/failed/locked）を表示", value=False)
if show_logs:
    st.info("各 pno フォルダに対する採否ログを確認します（`*_skip*.pdf` と side.json の ocr:{skipped, failed, locked} は除外）。")
    for pno in selected_pnos:
        st.markdown(f"#### 📁 {selected_shard}/{pno}")
        raw_pdfs = list_pdfs_in_pno(PDF_ROOT, selected_shard, pno)

        # （1）まず *_skip / side.json(skipped/failed/locked) を除外してログ
        filtered, skip_logs = _filter_skip(raw_pdfs)

        # （2）_ocr 優先ロジックを適用
        candidates, ocr_logs = decide_ocr_candidates(filtered)

        # ログ表示
        with st.expander(f"{pno} の採否ログ", expanded=True):
            for line in skip_logs:
                st.write(line)
            for line in ocr_logs:
                st.write(line)

        if not candidates:
            st.warning(f"{pno}: 取り込み対象の PDF がありません。")

# ============================================================
# 実行（取り込みパイプライン開始）— 進捗UI 強化版
# ============================================================
run = st.button("選択した pno フォルダ内の PDF を取り込み（ベクトルデータベースの作成）", type="primary")
if run:
    # 進捗UIコンテナ
    overall_progress = st.progress(0.0, text="準備中…")
    file_progress    = st.progress(0.0, text="ファイル進捗：待機中…")
    phase_box        = st.empty()
    live_stats_box   = st.container()
    log_box          = st.container()

    # フェーズ: 出力先準備
    t0 = time.time()
    phase_box.info("① 出力先を準備中…")
    vs_dir = ensure_vs_dir(VS_ROOT, backend, selected_shard)
    tracker = ProcessedFilesSimple(vs_dir / "processed_files.json")
    overall_progress.progress(0.05, text="出力先の準備完了（5%）")

    # フェーズ: 埋め込みバックエンド初期化
    phase_box.info("② 埋め込みバックエンドを初期化中…")
    estore = EmbeddingStore(backend=backend, openai_model=OPENAI_EMBED_MODEL)
    overall_progress.progress(0.10, text="バックエンド初期化（10%）")

    # フェーズ: ベクトルストアをロード
    phase_box.info("③ ベクトルストアをロード中…（大きい場合は時間がかかります）")
    t_vdb0 = time.time()
    vdb = NumpyVectorDB(vs_dir)  # 大規模だとここで時間がかかる
    t_vdb1 = time.time()
    overall_progress.progress(0.20, text=f"ベクトルストア読み込み完了（{t_vdb1 - t_vdb0:.1f}秒, 20%）")

    # meta の year 値
    try:
        year_val = int(selected_shard)
    except ValueError:
        year_val = None

    # ライブ統計表示（プレースホルダ）
    stat_cols = live_stats_box.columns(6)
    def render_stats(new_files, skipped_done, skipped_side, skipped_name, failed_files, add_chunks, i_done, i_total, start_time):
        elapsed = time.time() - start_time
        per = (i_done / i_total) if i_total else 0.0
        avg = (elapsed / max(i_done, 1))
        eta = (i_total - i_done) * avg
        stat_cols[0].write(f"**新規ファイル:** {new_files}")
        stat_cols[1].write(f"**既取込スキップ:** {skipped_done}")
        stat_cols[2].write(f"**side除外:** {skipped_side}")
        stat_cols[3].write(f"**_skip除外:** {skipped_name}")
        stat_cols[4].write(f"**失敗:** {failed_files}")
        stat_cols[5].write(f"**追加チャンク:** {add_chunks}")
        overall_progress.progress(
            min(0.20 + 0.80 * per, 1.0),
            text=f"処理中… {i_done}/{i_total} ファイル（経過 {int(elapsed)}秒 / 予測残り {_fmt_eta(eta)}）"
        )

    # 事前に対象のファイル総数を概算（候補列挙＋除外＋_ocr優先を合算）
    phase_box.info("④ 対象ファイルをスキャン中…")
    pno_to_candidates: dict[str, List[Tuple[Path, bool]]] = {}
    total_candidates = 0
    for pno in selected_pnos:
        raw_pdfs = list_pdfs_in_pno(PDF_ROOT, selected_shard, pno)
        filtered, _ = _filter_skip(raw_pdfs)
        candidates, _ = decide_ocr_candidates(filtered)
        pno_to_candidates[pno] = candidates
        total_candidates += len(candidates)
    overall_progress.progress(0.25, text=f"候補収集完了（{total_candidates}ファイル, 25%）")

    # カウンタ
    total_files_new = 0
    total_chunks    = 0
    skipped_done    = 0
    skipped_side    = 0
    skipped_name    = 0
    failed_files    = 0

    phase_box.info("⑤ ファイル取り込みを開始…")
    started = time.time()
    i_done = 0

    # pno ごとの ingest ループ
    for pno in selected_pnos:
        log_box.markdown(f"### 📁 プロジェクト: `{selected_shard}/{pno}`")
        # processed_files.json の旧キーを正準化
        migrate_processed_files_to_canonical(vs_dir / "processed_files.json", selected_shard, pno)

        candidates = pno_to_candidates.get(pno, [])
        file_progress.progress(0.0, text=f"{pno}: 0/{len(candidates)} 開始")

        for i_file, (pdf_path, ocr_flag) in enumerate(candidates, start=1):
            name = pdf_path.name
            key_full = f"{selected_shard}/{pno}/{name}"

            # 二重防御: _skip / side除外
            if _is_skip_file(pdf_path):
                skipped_name += 1
                log_box.info(f"⏭️ スキップ（_skip）: `{selected_shard}/{pno}` / **{name}**")
                i_done += 1
                file_progress.progress(i_file/len(candidates), text=f"{pno}: {i_file}/{len(candidates)}")
                render_stats(total_files_new, skipped_done, skipped_side, skipped_name, failed_files, total_chunks, i_done, total_candidates, started)
                continue

            is_ex, reason = _is_side_excluded(pdf_path)
            if is_ex:
                skipped_side += 1
                log_box.info(f"⏭️ スキップ（{reason}）: `{selected_shard}/{pno}` / **{name}**")
                i_done += 1
                file_progress.progress(i_file/len(candidates), text=f"{pno}: {i_file}/{len(candidates)}")
                render_stats(total_files_new, skipped_done, skipped_side, skipped_name, failed_files, total_chunks, i_done, total_candidates, started)
                continue

            # 既取込スキップ（後方互換キー対応）
            if tracker.is_done(key_full) or tracker.is_done(f"{selected_shard}/{name}") or tracker.is_done(name):
                skipped_done += 1
                log_box.info(f"⏭️ スキップ（既取込）: `{selected_shard}/{pno}` / **{name}**")
                i_done += 1
                file_progress.progress(i_file/len(candidates), text=f"{pno}: {i_file}/{len(candidates)}")
                render_stats(total_files_new, skipped_done, skipped_side, skipped_name, failed_files, total_chunks, i_done, total_candidates, started)
                continue

            # 実処理
            try:
                log_box.write(f"🚚 取り込み中: **{name}**（OCR優先={ocr_flag}）")
                t_f0 = time.time()
                add_files, add_chunks = ingest_pdf_file(
                    vdb=vdb,
                    estore=estore,
                    shard_id=selected_shard,
                    pno=pno,
                    pdf_path=pdf_path,
                    year_val=year_val,
                    batch_size=int(batch_size),
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                    ocr_flag=ocr_flag,
                    embed_model_label=(OPENAI_EMBED_MODEL if backend == "openai" else "local-model"),
                )
                t_f1 = time.time()
                tracker.mark_done(key_full)
                total_files_new += add_files
                total_chunks    += add_chunks
                log_box.success(f"✅ 完了: **{name}**（{add_chunks} チャンク, {t_f1 - t_f0:.1f}秒）")

            except Exception as e:
                failed_files += 1
                log_box.error(f"❌ 失敗: **{name}** - {e}")

            # 進捗更新
            i_done += 1
            file_progress.progress(i_file/len(candidates), text=f"{pno}: {i_file}/{len(candidates)}")
            render_stats(total_files_new, skipped_done, skipped_side, skipped_name, failed_files, total_chunks, i_done, total_candidates, started)

        # pnoサマリ
        log_box.caption(f"🔎 ベクトル総数（シャード内）: {get_vector_count(vs_dir):,d}")

    # 完了
    elapsed_all = time.time() - t0
    overall_progress.progress(1.0, text=f"完了（{elapsed_all:.1f}秒）")
    st.toast(
        f"✅ 完了: 新規 {total_files_new} ファイル / {total_chunks} チャンク（_ocr優先・side/_skip除外）",
        icon="✅",
    )
    
    # # 概算コスト（openai のみ）
    # if total_chunks > 0 and backend == "openai":
    #     st.markdown("### 💰 埋め込みコストの概算")
    #     total_tokens = 0
    #     meta_path = VS_ROOT / backend / selected_shard / "meta.jsonl"
    #     if meta_path.exists():
    #         with meta_path.open("r", encoding="utf-8") as f:
    #             for line in f:
    #                 try:
    #                     obj = json.loads(line)
    #                 except Exception:
    #                     continue
    #                 total_tokens += int(obj.get("chunk_len_tokens", 0))

    #     # lib.costs に統一
    #     est = estimate_embedding_cost(OPENAI_EMBED_MODEL, total_tokens)  # {"usd":..., "jpy":...}
    #     usd = float(est["usd"])
    #     jpy = float(est["jpy"])

    #     st.write(f"- モデル: **{OPENAI_EMBED_MODEL}**")
    #     st.write(f"- 総トークン数: {total_tokens:,}")
    #     st.write(f"- 概算コスト: `${usd:.4f}` ≈ ¥{jpy:,.0f}（為替 {DEFAULT_USDJPY:.2f} JPY/USD）")
    # elif total_chunks > 0:
    #     st.info("local backend のためコストは発生しません。")

    # === 概算コスト（openai のみ） ===
    if total_chunks > 0 and backend == "openai":
        from lib.costs import summarize_embedding_cost_from_meta, DEFAULT_USDJPY

        st.markdown("### 💰 埋め込みコストの概算（検証付き）")

        meta_path = VS_ROOT / backend / selected_shard / "meta.jsonl"
        summary = summarize_embedding_cost_from_meta(
            meta_path,
            model=OPENAI_EMBED_MODEL,
            rate=DEFAULT_USDJPY,
            include_source_paths=[pdf_path.name],  # 今回のPDF名で絞り込み
        )

        # ────── 概算表示 ──────
        st.write(f"- モデル: **{summary['model']}** (${summary['price_per_1M']:.3f} / 1M tok)")
        st.write(f"- チャンク数: {summary['n_chunks']:,}")
        st.write(f"- 総トークン数: {summary['total_tokens']:,}")
        st.write(
            f"- 概算コスト: `${summary['usd']:.4f}` ≈ ¥{summary['jpy']:,.0f} "
            f"（為替 {summary['rate']:.2f} JPY/USD）"
        )

        # ────── サニティ情報 ──────
        st.caption(
            f"Sanity: avg={summary['avg_tok']:,.0f} tok/chunk, "
            f"p95={summary['p95_tok']:,.0f}, max={summary['max_tok']:,.0f}"
        )

        # ────── 警告表示 ──────
        for w in summary["warnings"]:
            st.warning(w)

    elif total_chunks > 0:
        st.info("local backend のためコストは発生しません。")


