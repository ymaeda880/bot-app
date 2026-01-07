# pages/31_sample_pdfベクトル化.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import json
import time

# --- sys.path 調整（common_lib へ到達） ---
#（pageから）
import sys
_THIS = Path(__file__).resolve()
APP_ROOT = _THIS.parents[1]          # pages -> app root
PROJECTS_ROOT = _THIS.parents[3]     # auth_portal/pages -> projects/auth_portal
import sys
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

import streamlit as st

from lib.rag.rag_utils import EmbeddingStore, NumpyVectorDB, ProcessedFilesSimple
from lib.costs import summarize_embedding_cost_from_meta, DEFAULT_USDJPY

# 主要ロジックは lib 側へ分離（UI から関数を呼び出すだけ）
from lib.pdf_ingest import (
    OPENAI_EMBED_MODEL,
    decide_ocr_candidates,
    ingest_pdf_file,
    get_vector_count,
)

# from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie
# from common_lib.auth.auth_helpers import (
#     get_current_user_from_session_or_cookie,
#     is_admin,
#     clear_auth_caches,
# )

from common_lib.auth.auth_helpers import require_admin_user
# ============================================================
# 固定パス（要件）
# ============================================================
# PDF 置き場（固定）
PDF_ROOT = Path("/Volumes/Extreme SSD/RAG_data/sample/current")

# DB 置き場（固定）
# 例: database/data/vectorstore/openai_sample/current/
VS_BASE = Path("database/data/vectorstore/openai_sample")
SHARD_ID = "current"
BACKEND = "openai"  # 固定

# 仕様上の「pno」相当（lib 側の署名に合わせるため固定名を入れる）
PSEUDO_PNO = "sample"

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
        if _is_skip_file(p):
            logs.append(f"不採用: {p.name} — `_skip` のため除外")
            continue
        is_ex, reason = _is_side_excluded(p)
        if is_ex:
            logs.append(f"不採用: {p.name} — {reason}")
            continue
        kept.append(p)
    return kept, logs

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
# ページ基本設定
# ============================================================
st.set_page_config(
    page_title="Sample PDF ベクトル化（openai固定）",
    page_icon="📘",
    layout="wide",
)

# ============================================================
# アクセス制御（管理者チェック）※ 65_ログ管理.py と同方式
# ============================================================

sub = require_admin_user(st)
if not sub:
    st.error("🚫 このページは管理者のみアクセスできます。")
    st.stop()

user = sub
   
# ============================================================
# タイトル＋ログインバッジ（65_ログ管理.py と同スタイル）
# ============================================================
col_title, col_user = st.columns([5, 2], vertical_alignment="center")

with col_title:
    st.title("📘 規定集（sample）のベクトル化")
    st.write("（openai固定 / shard=current固定）")

with col_user:
    # ここは 65 と同じ呼び方
    #user, payload = get_current_user_from_session_or_cookie(st)
    if user:
        st.success(f"管理者としてログイン中: **{user}**")
    else:
        st.warning("未ログイン")

with st.expander("📘 対象・除外ルール（クリックで表示）", expanded=False):
    st.markdown(
    """
**対象:**
- PDF 置き場: `/ssd/RAG_data/sample/current/`
- DB 置き場: `/database/data/vectorstore/openai_sample/current/`
- backend: **openai 固定**
- shard: **current 固定**

**除外:**
- `*_skip*.pdf` は除外
- `<basename>_side.json` の `ocr` が `skipped/failed/locked` は除外
- `_ocr.pdf` と素の PDF が同居する場合は **_ocr を優先**
"""
)

# ------------------------------------------------------------
# サイドバー（表示のみ）
# ------------------------------------------------------------
with st.sidebar:
    st.subheader("📌 固定設定（表示のみ）")
    st.text_input("PDF_ROOT", str(PDF_ROOT), disabled=True)
    st.text_input("VS_BASE", str(VS_BASE), disabled=True)
    st.text_input("SHARD", SHARD_ID, disabled=True)
    st.text_input("BACKEND", BACKEND, disabled=True)

# ============================================================
# 取り込みパラメータ（chunking / batching のみ）
# ============================================================
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.caption(f"🔧 Embedding モデルは **{OPENAI_EMBED_MODEL}（次元は埋め込み結果に従う）**")
with col2:
    chunk_size = st.number_input("チャンクサイズ（文字）", 200, 3000, 900, 50)
    overlap    = st.number_input("オーバーラップ（文字）", 0, 600, 150, 10)
with col3:
    batch_size = st.number_input("埋め込みバッチ数", 8, 512, 64, 8)

st.info(f"**入力**: `{PDF_ROOT}/<folder>/*.pdf`\n\n**出力**: `{VS_BASE}/{SHARD_ID}`（openai固定）")


# ============================================================
# 入力チェック
# ============================================================
if not PDF_ROOT.exists():
    st.error(f"PDF_ROOT が存在しません: {PDF_ROOT}")
    st.stop()

subdirs = sorted([d for d in PDF_ROOT.iterdir() if d.is_dir()])
if not subdirs:
    st.warning(f"{PDF_ROOT} にサブフォルダがありません。例: {PDF_ROOT}/<folder>/*.pdf")
    st.stop()

# サブフォルダ直下のPDFを列挙（再帰なし）
raw_pdfs: List[Path] = []
for d in subdirs:
    raw_pdfs.extend([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
raw_pdfs = sorted(raw_pdfs)

if not raw_pdfs:
    st.warning(f"{PDF_ROOT}/<folder> 配下に PDF がありません。")
    st.stop()


# ============================================================
# 取り込み前プレビュー（採否ログ）
# ============================================================
st.markdown("#### 🔎 取り込み候補プレビュー")
filtered, skip_logs = _filter_skip(raw_pdfs)
candidates, ocr_logs = decide_ocr_candidates(filtered)

st.write(f"- PDF総数: {len(raw_pdfs)}")
st.write(f"- 除外後: {len(filtered)}")
st.write(f"- _ocr優先後の候補: **{len(candidates)}**")

with st.expander("🧾 採否ログ（_skip / side.json / _ocr 優先）", expanded=False):
    for line in skip_logs:
        st.write(line)
    for line in ocr_logs:
        st.write(line)

# 候補一覧（長ければ折りたたみ）
with st.expander("候補PDF一覧", expanded=False):
    st.code("\n".join([p.name for p, _ in candidates]), language="text")

# ============================================================
# 実行
# ============================================================
run = st.button("📘 規定集PDFを取り込み（ベクトルDB作成）", type="primary")
if run:
    overall_progress = st.progress(0.0, text="準備中…")
    file_progress    = st.progress(0.0, text="ファイル進捗：待機中…")
    phase_box        = st.empty()
    live_stats_box   = st.container()
    log_box          = st.container()

    t0 = time.time()

    # ① 出力先準備
    phase_box.info("① 出力先を準備中…")
    vs_dir = VS_BASE / SHARD_ID
    vs_dir.mkdir(parents=True, exist_ok=True)
    # ↑ ensure_vs_dir の仕様が `VS_ROOT/<backend>/<shard>` なら、
    # VS_BASE を “VS_ROOT として扱う” 前提で、VS_BASE=openai_sample を渡しています。
    tracker = ProcessedFilesSimple(vs_dir / "processed_files.json")
    overall_progress.progress(0.05, text="出力先の準備完了（5%）")

    # ② openai backend 初期化（固定）
    phase_box.info("② 埋め込みバックエンド（openai）を初期化中…")
    estore = EmbeddingStore(backend="openai", openai_model=OPENAI_EMBED_MODEL)
    overall_progress.progress(0.10, text="バックエンド初期化（10%）")

    # ③ ベクトルストア読み込み
    phase_box.info("③ ベクトルストアをロード中…")
    t_vdb0 = time.time()
    vdb = NumpyVectorDB(vs_dir)
    t_vdb1 = time.time()
    overall_progress.progress(0.20, text=f"ベクトルストア読み込み完了（{t_vdb1 - t_vdb0:.1f}秒, 20%）")

    # ライブ統計
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

    # カウンタ
    total_files_new = 0
    total_chunks    = 0
    skipped_done    = 0
    skipped_side    = 0
    skipped_name    = 0
    failed_files    = 0

    processed_meta_keys: set[str] = set()
    processed_filenames: set[str] = set()

    phase_box.info("④ ファイル取り込みを開始…")
    started = time.time()

    total_candidates = len(candidates)
    i_done = 0

    file_progress.progress(0.0, text=f"0/{total_candidates} 開始")

    for i_file, (pdf_path, ocr_flag) in enumerate(candidates, start=1):
       
        name = pdf_path.name
        pno = pdf_path.parent.name  # ★ サブフォルダ名を pno として扱う
        key_full = f"{SHARD_ID}/{pno}/{name}"

        # 二重防御
        if _is_skip_file(pdf_path):
            skipped_name += 1
            log_box.info(f"⏭️ スキップ（_skip）: **{name}**")
            i_done += 1
            file_progress.progress(i_file/total_candidates, text=f"{i_file}/{total_candidates}")
            render_stats(total_files_new, skipped_done, skipped_side, skipped_name, failed_files, total_chunks, i_done, total_candidates, started)
            continue

        is_ex, reason = _is_side_excluded(pdf_path)
        if is_ex:
            skipped_side += 1
            log_box.info(f"⏭️ スキップ（{reason}）: **{name}**")
            i_done += 1
            file_progress.progress(i_file/total_candidates, text=f"{i_file}/{total_candidates}")
            render_stats(total_files_new, skipped_done, skipped_side, skipped_name, failed_files, total_chunks, i_done, total_candidates, started)
            continue

        # 既取込
        if tracker.is_done(key_full) or tracker.is_done(name):
            skipped_done += 1
            log_box.info(f"⏭️ スキップ（既取込）: **{name}**")
            i_done += 1
            file_progress.progress(i_file/total_candidates, text=f"{i_file}/{total_candidates}")
            render_stats(total_files_new, skipped_done, skipped_side, skipped_name, failed_files, total_chunks, i_done, total_candidates, started)
            continue

        # 実処理
        try:
            log_box.write(f"🚚 取り込み中: **{name}**（OCR優先={ocr_flag}）")
            t_f0 = time.time()

            add_files, add_chunks = ingest_pdf_file(
                vdb=vdb,
                estore=estore,
                shard_id=SHARD_ID,
                pno=pno,
                pdf_path=pdf_path,
                year_val=None,  # regulations/sample は年度概念不要
                batch_size=int(batch_size),
                chunk_size=int(chunk_size),
                overlap=int(overlap),
                ocr_flag=ocr_flag,
                embed_model_label=OPENAI_EMBED_MODEL,
            )

            t_f1 = time.time()
            tracker.mark_done(key_full)
            total_files_new += add_files
            total_chunks    += add_chunks

            processed_meta_keys.add(key_full)
            processed_filenames.add(name)

            log_box.success(f"✅ 完了: **{name}**（{add_chunks} チャンク, {t_f1 - t_f0:.1f}秒）")

        except Exception as e:
            failed_files += 1
            log_box.error(f"❌ 失敗: **{name}** - {e}")

        i_done += 1
        file_progress.progress(i_file/total_candidates, text=f"{i_file}/{total_candidates}")
        render_stats(total_files_new, skipped_done, skipped_side, skipped_name, failed_files, total_chunks, i_done, total_candidates, started)

    # 完了
    elapsed_all = time.time() - t0
    overall_progress.progress(1.0, text=f"完了（{elapsed_all:.1f}秒）")
    st.toast(
        f"✅ 完了: 新規 {total_files_new} ファイル / {total_chunks} チャンク（_ocr優先・side/_skip除外）",
        icon="✅",
    )

    log_box.caption(f"🔎 ベクトル総数（current）: {get_vector_count(vs_dir):,d}")

    # === 概算コスト（openaiのみ） ===
    if total_chunks > 0:
        st.markdown("### 💰 埋め込みコストの概算（検証付き）")

        meta_path = vs_dir / "meta.jsonl"
        include_list = sorted(processed_meta_keys | processed_filenames)

        summary = summarize_embedding_cost_from_meta(
            meta_path,
            model=OPENAI_EMBED_MODEL,
            rate=DEFAULT_USDJPY,
            include_source_paths=include_list,
        )

        st.write(f"- モデル: **{summary['model']}** (${summary['price_per_1M']:.3f} / 1M tok)")
        st.write(f"- チャンク数: {summary['n_chunks']:,}")
        st.write(f"- 総トークン数: {summary['total_tokens']:,}")
        st.write(
            f"- 概算コスト: `${summary['usd']:.4f}` ≈ ¥{summary['jpy']:,.0f} "
            f"（為替 {summary['rate']:.2f} JPY/USD）"
        )
        st.caption(
            f"Sanity: avg={summary['avg_tok']:,.0f} tok/chunk, "
            f"p95={summary['p95_tok']:,.0f}, max={summary['max_tok']:,.0f}"
        )

        for w in summary["warnings"]:
            st.warning(w)
