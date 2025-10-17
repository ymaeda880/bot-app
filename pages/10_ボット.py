# pages/10_ボット.py
# =============================================================================
# 💬 Internal Bot (RAG, Shards) — year/pno フィルタ付き（候補プリフィルタ＋TopKブースト）
# + ⏱ 実行タイミング計測（VectorDB 走査 / 埋め込み / 検索 / GPT API / ストリーム）
# =============================================================================

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable, Set
import heapq
from itertools import count
import json

import streamlit as st
import numpy as np

from config.path_config import PATHS
from config.sample_questions import SAMPLES, ALL_SAMPLES
from lib.text_normalize import normalize_ja_text
from lib.prompts.bot_prompt import build_prompt
from lib.gpt_responder import GPTResponder, CompletionResult
from lib.rag.rag_utils import EmbeddingStore, NumpyVectorDB
from lib.costs import (
    MODEL_PRICES_USD, EMBEDDING_PRICES_USD, DEFAULT_USDJPY,
    ChatUsage, estimate_chat_cost, estimate_embedding_cost, usd_to_jpy,
    render_usage_summary, _model_prices_per_1k
)
from lib.openai_utils import count_tokens
from lib.bot_utils import (
    list_shard_dirs_openai, norm_path, fmt_source,
    enrich_citations, citation_tag, get_openai_api_key,
    parse_inline_files, strip_inline_files,
    # 新規ユーティリティ
    to_halfwidth_digits, clean_pno_token, parse_years, parse_pnos, norm_pno_forms,
    year_ok, pno_ok, file_ok, scan_candidate_files, filters_caption,
)

from lib.metrics.timing_utils import Timings, stream_with_timing, render_metrics_ui
import datetime as dt
import time


# from lib.ui import warn_before_close
# warn_before_close()  # ←必要なら有効化

# ===== 価格テーブル整形（USD/1K tok） =======================================
MODEL_PRICES_PER_1K = _model_prices_per_1k()

VS_ROOT: Path = PATHS.vs_root

# ===== Streamlit 基本設定／タイトル ==========================================
st.set_page_config(page_title="Chat Bot (Sharded)", page_icon="💬", layout="wide")
st.title("💬 Internal Bot (RAG, Shards)")

if "q" not in st.session_state:
    st.session_state.q = ""

def _set_q(x: str) -> None:
    st.session_state.q = x or ""

####################
# メモリ監視（任意）
#####################
from lib.monitors.ui_memory_monitor import render_memory_kpi_row

st.divider()
st.markdown("### 🧠 メモリ状況（参考）")
render_memory_kpi_row()

# ===== サイドバー ==============================================================
with st.sidebar:
    st.header("設定")

    model_candidates = [m for m in MODEL_PRICES_PER_1K if m.startswith(("gpt-5", "gpt-4.1"))]
    all_models_sorted = sorted(model_candidates, key=lambda x: (0 if x.startswith("gpt-5") else 1, x))
    if not all_models_sorted:
        st.error("利用可能な Responses モデルが見つかりません。lib/costs.MODEL_PRICES_USD を確認してください。")
        st.stop()
    default_idx = all_models_sorted.index("gpt-5-mini") if "gpt-5-mini" in all_models_sorted else 0
    chat_model = st.selectbox("モデル（Responses）", all_models_sorted, index=default_idx)

    top_k = st.slider("検索件数（Top-K）", 1, 12, 6, 1)

    detail = {
        "簡潔": "concise", "標準": "standard", "詳細": "detailed", "超詳細": "very_detailed"
    }[st.selectbox("詳しさ", ["簡潔", "標準", "詳細", "超詳細"], index=2)]

    cite = st.checkbox("出典を [S1] で促す", True)
    max_tokens = st.slider("最大出力トークン", 1000, 40000, 12000, 500)
    answer_backend = st.radio("回答生成", ["OpenAI", "Retrieve-only"], index=0)
    sys_inst = st.text_area("System Instruction", "あなたは優秀な社内のアシスタントです.", height=80)
    display_mode = st.radio("表示モード", ["逐次表示（ストリーム）", "一括表示"], index=0)

    st.divider(); st.subheader("検索対象シャード（OpenAI）")
    shard_dirs_all = list_shard_dirs_openai(VS_ROOT)
    shard_ids_all = [p.name for p in shard_dirs_all]
    target_shards = st.multiselect("（未選択=すべて）", shard_ids_all, default=shard_ids_all)

    # ✅ year/pno 入力（パースは bot_utils に委譲）
    st.divider(); st.subheader("year / pno フィルタ")
    years_input = st.text_input("year（カンマ区切り・任意）", value="", help="例: 2019,2023")
    pnos_input  = st.text_input("pno（プロジェクト番号, カンマ区切り・任意）", value="", help="例: 010,045,120")

    years_sel: Set[int] = parse_years(years_input)
    pnos_raw: Set[str]  = parse_pnos(pnos_input)

    # pno の表記ゆれ（元/ゼロ除去/3桁ゼロ埋め）を吸収
    pnos_sel_norm: Set[str] = set()
    for p in pnos_raw:
        pnos_sel_norm |= norm_pno_forms(p)

    st.caption(
        f"year フィルタ: {sorted(list(years_sel)) or '（未指定）'} / "
        f"pno フィルタ: {sorted(list(pnos_sel_norm)) or '（未指定）'}"
    )
    st.caption("※ 未入力なら全件対象。どちらか/両方を入力した場合のみフィルタします。")

    st.caption("参照ファイルを追加（例: 2025/foo.pdf, 2024/bar.pdf）")
    file_whitelist_str = st.text_input("参照ファイル（任意）", value="")
    ui_file_whitelist = {s.strip() for s in file_whitelist_str.split(",") if s.strip()}

    api_key = get_openai_api_key()
    if not api_key and answer_backend == "OpenAI":
        st.error("OpenAI APIキーが secrets.toml / 環境変数にありません。")

    st.divider(); st.markdown("### 📂 解決パス（参照用）")
    st.text_input("VS_ROOT", str(PATHS.vs_root), disabled=True)
    st.text_input("PDF_ROOT", str(PATHS.pdf_root), disabled=True)
    st.text_input("BACKUP_ROOT", str(PATHS.backup_root), disabled=True)
    if hasattr(PATHS, "data_root"):
        st.text_input("DATA_ROOT", str(PATHS.data_root), disabled=True)

    st.divider(); st.subheader("🧪 デモ用サンプル質問")
    cat = st.selectbox("カテゴリを選択", ["（未選択）"] + list(SAMPLES.keys()))
    sample_options = ["（未選択）"] if cat == "（未選択）" else ["（未選択）"] + SAMPLES.get(cat, [])
    sample = st.selectbox("サンプル質問を選択", sample_options, index=0)
    cols_demo = st.columns(2)
    with cols_demo[0]:
        st.button("⬇️ この質問を入力欄へセット", width='stretch',
                  disabled=(sample in ("", "（未選択）")), on_click=lambda: _set_q(sample))
    with cols_demo[1]:
        st.button("🎲 ランダム挿入", width='stretch',
                  on_click=lambda: _set_q(str(np.random.choice(ALL_SAMPLES)) if ALL_SAMPLES else ""))
    send_now = st.button("🚀 サンプルで即送信", width='stretch',
                         disabled=(st.session_state.q.strip() == ""))
    
    #### メモリ状況（回答前 / 回答後）
    st.divider()
    st.subheader("🧠 メモリ状況（回答前 / 回答後）")
    # 回答前/後のスナップショット描画先（この順序で下に並ぶ）
    mem_pre_box = st.container()
    mem_post_box = st.container()

    
    


# ===== 本文 =========================================================
q = st.text_area("質問を入力", value=st.session_state.q, height=100,
                 placeholder="この社内ボットに質問してください…")
if q != st.session_state.q:
    st.session_state.q = q

go = st.button("送信", type="primary")
go = go or bool(locals().get("send_now"))

# ===== 実行：検索 → 生成 → コスト → 参照 ===========================
if go and st.session_state.q.strip():
    timings = Timings()
    timings.mark("pipeline_start")

    # ← ここが“回答前”。この時点のメモリを上段にレンダ
    with mem_pre_box:
        st.caption("（回答前スナップショット）")
        render_memory_kpi_row()

    try:
        vs_backend_dir = PATHS.vs_root / "openai"
        if not vs_backend_dir.exists():
            st.warning(f"ベクトルが見つかりません（{vs_backend_dir}）。先にベクトル化を実行してください。")
            st.stop()

        selected_ids = target_shards or [p.name for p in list_shard_dirs_openai(PATHS.vs_root)]
        shard_dirs = [vs_backend_dir / s for s in selected_ids]
        shard_dirs = [p for p in shard_dirs if p.is_dir() and (p / "vectors.npy").exists()]
        if not shard_dirs:
            st.warning("検索可能なシャードがありません。")
            st.stop()

        # インライン [[files: ...]] と UI ホワイトリストを合算（正規化は file_ok 内で再利用）
        inline_files = parse_inline_files(st.session_state.q)
        effective_whitelist_norm: Set[str] = {norm_path(x) for x in (ui_file_whitelist | inline_files)}

        # ---- meta.jsonl 走査（候補抽出） ----
        timings.mark("scan_start")
        cand_by_shard, cand_total = scan_candidate_files(shard_dirs, years_sel, pnos_sel_norm)
        timings.mark("scan_end")

        if (years_sel or pnos_sel_norm) and cand_total == 0:
            st.warning("指定の year/pno に一致する**候補ファイルが 0 件**のため、検索を実行しません。条件を見直してください。")
            st.stop()

        # year/pno 由来の候補をホワイトリストに合成
        for _sd, files in cand_by_shard.items():
            effective_whitelist_norm |= files

        # デバッグ表記
        st.caption(filters_caption(years_sel, pnos_sel_norm, cand_total if (years_sel or pnos_sel_norm) else None))

        # ---- クエリ埋め込み ----
        question = normalize_ja_text(strip_inline_files(st.session_state.q))
        estore = EmbeddingStore(backend="openai")
        question_tokens = count_tokens(st.session_state.q, "text-embedding-3-large")

        timings.mark("embed_start")
        qv = estore.embed([question]).astype("float32")
        timings.mark("embed_end")

        # ---- 各シャード TopK をマージ ----
        timings.mark("search_start")
        K = int(top_k)
        heap_: List[Tuple[float, int, int, Dict[str, Any]]] = []  # (score, tiebreak, row_idx, meta)
        tie = count()
        for shp in shard_dirs:
            try:
                vdb = NumpyVectorDB(shp)
                local_k = K
                if years_sel or pnos_sel_norm or effective_whitelist_norm:
                    local_k = max(K * 10, 50)
                hits = vdb.search(qv, top_k=local_k, return_="similarity")
                for h in hits:
                    if isinstance(h, tuple) and len(h) == 3:
                        row_idx, score, meta = h
                    else:
                        score, meta = h
                        row_idx = -1

                    md = dict(meta or {}); md["shard_id"] = shp.name

                    # 統一判定（year/pno/file）
                    if not year_ok(md, years_sel):                 # 年
                        continue
                    if not pno_ok(md, pnos_sel_norm):              # プロジェクト番号
                        continue
                    if not file_ok(md, effective_whitelist_norm):  # ファイル限定
                        continue

                    sc = float(score)
                    if len(heap_) < K:
                        heapq.heappush(heap_, (sc, next(tie), row_idx, md))
                    elif sc > heap_[0][0]:
                        heapq.heapreplace(heap_, (sc, next(tie), row_idx, md))
            except Exception as e:
                st.warning(f"シャード {shp.name} の検索でエラー: {e}")
        timings.mark("search_end")

        raw_hits = [(rid, sc, md) for (sc, _tb, rid, md) in sorted(heap_, key=lambda x: x[0], reverse=True)]
        if not raw_hits:
            if years_sel or pnos_sel_norm:
                st.warning("条件には合う候補ファイルがありましたが、上位スコアに入りませんでした。\n"
                           "👉 Top-K を増やす／クエリを具体化する／シャードを絞る、のいずれかを試してください。")
            else:
                st.warning("該当コンテキストが見つかりませんでした。")
            st.stop()

        # ---- 回答生成 ----
        chat_prompt_tokens = chat_completion_tokens = 0
        use_backend = "OpenAI" if (answer_backend == "OpenAI" and api_key) else "Retrieve-only"

        if use_backend == "OpenAI":
            labeled = [
                f"[S{i}] {m.get('text', '')}\n[meta: {fmt_source(m)} / score={float(s):.3f}]"
                for i, (_rid, s, m) in enumerate(raw_hits, 1)
            ]
            prompt = build_prompt(
                question, labeled, sys_inst=sys_inst, style_hint=detail, cite=cite, strict=False
            )
            responder = GPTResponder(api_key=api_key)
            use_stream = (display_mode == "逐次表示（ストリーム）")

            # 🔹 逐次出力（ストリーム）
            if use_stream:
                st.subheader("🧠 回答（逐次表示）")
                timings.mark("gpt_req_start")
                with st.chat_message("assistant"):
                    st.write_stream(
                        stream_with_timing(
                            responder.stream(
                                model=chat_model,
                                system_instruction=sys_inst,
                                user_content=prompt,
                                max_output_tokens=int(max_tokens),
                                on_error_text="Responses stream error."
                            ),
                            timings,
                            first_key="gpt_first_token",
                            done_key="gpt_done",
                        )
                    )
                answer = responder.final_text or ""
                chat_prompt_tokens = responder.usage.input_tokens
                chat_completion_tokens = responder.usage.output_tokens
                # 念のためのフォールバック
                timings.marks.setdefault("gpt_first_token", timings.marks.get("gpt_req_start"))
                timings.marks.setdefault("gpt_done", dt.datetime.now())
                timings.perf.setdefault("gpt_first_token", timings.perf.get("gpt_req_start", time.perf_counter()))
                timings.perf.setdefault("gpt_done", time.perf_counter())

            # 🔹 一括表示モード
            else:
                st.subheader("🧠 回答（一括表示）")
                timings.mark("gpt_req_start")
                with st.spinner("回答生成中…"):
                    result: CompletionResult = responder.complete(
                        model=chat_model,
                        system_instruction=sys_inst,
                        user_content=prompt,
                        max_output_tokens=int(max_tokens)
                    )
                # 一括は「最初のトークン＝完了」とみなす
                timings.mark("gpt_first_token")
                timings.mark("gpt_done")

                answer = result.text or ""
                chat_prompt_tokens = result.usage.input_tokens
                chat_completion_tokens = result.usage.output_tokens

                st.write(enrich_citations(answer, raw_hits))

            # ---- 出典処理（両モード共通）----
            answer = enrich_citations(answer, raw_hits)
            import re as _re
            citations = _re.findall(r"\[S[^\]]+\]", answer)
            if citations:
                seen = []
                for c in citations:
                    if c not in seen:
                        seen.append(c)
                with st.expander("📝 出典拡張済み最終テキスト", expanded=False):
                    st.caption("以下は回答内の出典タグを整理した一覧です。")
                    st.markdown("### 📚 出典（出典ごとに改行）")
                    st.text("\n".join(seen))
            else:
                st.caption("出典タグは検出されませんでした。")

        else:
            st.subheader("🧩 取得のみ（要約なし）")
            st.info("Retrieve-only モードです。下の参照コンテキストをご覧ください。")
            answer = ""  # 念のため

        # ---- 参照コンテキスト ----
        with st.expander("🔍 参照コンテキスト（上位ヒット）", expanded=False):
            for i, (_rid, score, meta) in enumerate(raw_hits, 1):
                txt = str(meta.get("text", "") or "")
                label = fmt_source(meta)
                year = meta.get("year", "")
                pno = meta.get("pno", "") or meta.get("project_no", "")
                snippet = (txt[:1000] + "…") if len(txt) > 1000 else txt
                st.markdown(
                    f"**[{citation_tag(i, meta)}] score={float(score):.3f}**  "
                    f"`{label}` — **year:** {year} / **pno:** {pno}\n\n{snippet}"
                )

        # ---- 使用量の概算 ----
        render_usage_summary(
            embedding_model="text-embedding-3-large",
            embedding_tokens=question_tokens,
            chat_model=chat_model,
            chat_prompt_tokens=chat_prompt_tokens,
            chat_completion_tokens=chat_completion_tokens,
            use_backend_openai=(use_backend == "OpenAI"),
            title="📊 使用量の概算",
        )
        # ⏱ パイプライン終了
        timings.mark("pipeline_end")
        render_metrics_ui(timings)

        # ⏱ パイプライン終了直前/直後に“回答後”のメモリを下段にレンダ
        with mem_post_box:
            st.caption("（回答後スナップショット）")
            render_memory_kpi_row()

    except Exception as e:
        st.error(f"検索/生成中にエラー: {e}")
        # 例外時も可能な範囲でタイムラインを表示
        timings.marks.setdefault("pipeline_end", dt.datetime.now())
        timings.perf.setdefault("pipeline_end", time.perf_counter())
        render_metrics_ui(timings)
else:
    st.info("質問を入力して『送信』を押してください。サイドバーで設定できます。")

# 末尾：メモリ監視（任意で二重表示を避けるためコメントアウト例）
# st.divider()
# st.markdown("### 🧠 メモリ状況（参考）")
# render_memory_kpi_row()
