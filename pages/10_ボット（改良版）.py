# pages/10_ボット（改良版）.py
# ============================================
# gpt-5 / gpt-4.1（Responses API 専用）
# - 逐次表示（ストリーミング）⇔ 一括表示 切替
# - ストリーミング不可時は自動で一括表示にフォールバック
# - max_output_tokens / usage.input_tokens, output_tokens 対応
# - GPT呼び出しは lib/gpt_responder に集約
# ============================================

"""
Internal Bot (RAG, Shards) — Streamlit アプリ

本モジュールは、社内向けのRAGチャットボットUIを提供します。
- ベクトルストア（OpenAI埋め込み）から文書シャードを横断検索（Top-Kマージ）
- OpenAI Responses API で回答生成（gpt-5 / gpt-4.1 系想定）
- 逐次表示（ストリーミング）と一括表示をUIから切り替え可能
- ストリーミングが組織未検証等で禁止される場合は、自動で一括表示にフォールバック
- コスト概算（埋め込み・生成）を表示
- 上位参照コンテキスト（S1..）も確認可能

NOTE:
- 実運用では secrets.toml か環境変数で OPENAI_API_KEY を渡してください。
- ベクトルストアは OpenAI 固定。Numpy ベースの簡易VDBに対して検索します。
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import random
import heapq
from itertools import count
import unicodedata
import re

import streamlit as st
import numpy as np

# ★ GPT呼び出しはこのラッパに一元化
from lib.gpt_responder import GPTResponder, CompletionResult, Usage

from lib.rag_utils import EmbeddingStore, NumpyVectorDB
from config.path_config import PATHS
from lib.text_normalize import normalize_ja_text
from config.sample_questions import SAMPLES, ALL_SAMPLES

from lib.costs import (
    MODEL_PRICES_USD, EMBEDDING_PRICES_USD, DEFAULT_USDJPY,
    ChatUsage, estimate_chat_cost, estimate_embedding_cost, usd_to_jpy,
)

from lib.prompts.bot_prompt import build_prompt

# ========= /1M → /1K 変換（表示用） =========
MODEL_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    # モデル単価テーブル（USD/1M tokens）を表示/計算用に USD/1K tokens に変換
    m: {"in": float(p.get("in", 0.0)) / 1000.0, "out": float(p.get("out", 0.0)) / 1000.0}
    for m, p in MODEL_PRICES_USD.items()
}

# ========= パス =========
VS_ROOT: Path = PATHS.vs_root  # 例: <project>/data/vectorstore（OpenAIバックエンド配下にシャード格納）


# ========= ユーティリティ =========
def _count_tokens(text: str, model_hint: str = "cl100k_base") -> int:
    """
    テキストのおおよそのトークン数を見積もる。

    - tiktoken が使える場合はモデル名ヒントからエンコーディングを取得して精度高くカウント
    - 使えない場合は 1トークン ≒ 4文字 の近似で概算

    Args:
        text (str): 対象テキスト
        model_hint (str): tiktoken のモデル名ヒント（既定: cl100k_base）

    Returns:
        int: 概算トークン数（最低1）
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_hint)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))


def _fmt_source(meta: Dict[str, Any]) -> str:
    """
    検索ヒットのメタ情報から、人間可読なソース表記を作る。

    例: "foo.pdf p.12 (chunk_id)" または "foo.pdf" / "(unknown)"

    Args:
        meta (Dict[str, Any]): メタデータ辞書（file/page/chunk_idなど）

    Returns:
        str: 整形済みのラベル文字列
    """
    f = str(meta.get("file", "") or "")
    p = meta.get("page", None)
    cid = str(meta.get("chunk_id", "") or "")
    base = f"{f} p.{int(p)}" if (f and p is not None) else (f or "(unknown)")
    return f"{base} ({cid})" if cid else base


def _list_shard_dirs_openai() -> List[Path]:
    """
    OpenAI バックエンド用のベクトルストア直下にあるシャードディレクトリを列挙。

    Returns:
        List[Path]: シャードディレクトリのリスト（存在しない場合は空）
    """
    base = VS_ROOT / "openai"
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def _norm_path(s: str) -> str:
    """
    ファイル名/パスの比較用に正規化（NFKC, トリム, セパレータ統一, 小文字化）。

    Args:
        s (str): 入力パス文字列

    Returns:
        str: 正規化後のパス
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().replace("\\", "/")
    return s.lower()


def _get_openai_api_key() -> str | None:
    """
    OpenAI API キーを secrets.toml（st.secrets.openai.api_key） → 環境変数 の順で取得。

    Returns:
        Optional[str]: 見つかればキー文字列、なければ None
    """
    try:
        ok = st.secrets.get("openai", {}).get("api_key")
        if ok:
            return str(ok)
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


# ---------- モデル候補（Responses専用） ----------
RESPONSES_MODELS = [
    m for m in MODEL_PRICES_PER_1K.keys() if m.startswith("gpt-5") or m.startswith("gpt-4.1")
]


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Chat Bot (Sharded)", page_icon="💬", layout="wide")
st.title("💬 Internal Bot (RAG, Shards)")

# 入力テキストのセッション保持（サンプル挿入と連動）
if "q" not in st.session_state:
    st.session_state.q = ""


def _set_question(text: str):
    """サンプル選択ボタンから入力欄へテキストを流し込むためのコールバック。"""
    st.session_state.q = text


with st.sidebar:
    st.header("設定")

    # ---- モデル選択（gpt-5 系を先頭に来るよう並び順工夫）----
    st.markdown("### 回答モデル（Responses API）")
    all_models_sorted = sorted(
        RESPONSES_MODELS, key=lambda x: (0 if x.startswith("gpt-5") else 1, x)
    )
    default_idx = all_models_sorted.index("gpt-5-mini") if "gpt-5-mini" in all_models_sorted else 0
    chat_model = st.selectbox("モデルを選択", all_models_sorted, index=default_idx)

    # ---- ベクトル検索の Top-K ----
    top_k = st.slider("検索件数（Top-K）", 1, 12, 6, 1)

    # ---- 回答スタイル（プロンプト側に渡すヒント値）----
    label_to_value = {"簡潔": "concise", "標準": "standard", "詳細": "detailed", "超詳細": "very_detailed"}
    detail_label = st.selectbox("詳しさ", list(label_to_value.keys()), index=2)
    detail = label_to_value[detail_label]

    # ---- 引用表記（[S1] など）を促すかどうか ----
    cite = st.checkbox("出典を角括弧で引用（[S1] 等）", value=True)

    # ---- 出力トークン上限（Responses API の max_output_tokens）----
    max_tokens = st.slider("最大出力トークン（目安）", 1000, 40000, 12000, 500)

    # ---- 回答生成バックエンド（OpenAI / 取得のみ）----
    answer_backend = st.radio("回答生成", ["OpenAI", "Retrieve-only"], index=0)

    # ---- System Instruction（役割規定）----
    sys_inst = st.text_area("System Instruction", "あなたは優秀な社内のアシスタントです.", height=80)

    # ---- 表示モード（逐次表示 or 一括表示）----
    display_mode = st.radio(
        "表示モード",
        ["逐次表示（ストリーミング）", "一括表示"],
        index=0,
        help="逐次表示だと体感が速くなります。Responses API の stream を使用。"
    )

    # ---- 検索対象シャードの絞り込み（OpenAI 埋め込み専用）----
    st.divider()
    st.subheader("検索対象シャード（OpenAI）")
    shard_dirs_all = _list_shard_dirs_openai()
    shard_ids_all = [p.name for p in shard_dirs_all]
    target_shards = st.multiselect("（未選択=すべて）", shard_ids_all, default=shard_ids_all)

    # ---- 参照ファイルのホワイトリスト（年/ファイル名）----
    st.caption("特定ファイルだけで検索したい場合: 年/ファイル名 をカンマ区切り（例: 2025/foo.pdf, 2024/bar.pdf）")
    file_whitelist_str = st.text_input("参照ファイル（任意）", value="")
    file_whitelist = {s.strip() for s in file_whitelist_str.split(",") if s.strip()}

    # ---- APIキー存在チェック（UI側に警告表示）----
    has_key = bool(_get_openai_api_key())
    if not has_key and answer_backend == "OpenAI":
        st.error("OpenAI APIキーが secrets.toml / 環境変数にありません。埋め込みと回答生成の双方に必須です。")

    # ---- 解決済みの主要ディレクトリを確認（デバッグ/サポート向け）----
    st.divider()
    st.markdown("### 📂 解決パス（参照用）")
    st.text_input("VS_ROOT", str(PATHS.vs_root), key="p_vs", disabled=True)
    st.text_input("PDF_ROOT", str(PATHS.pdf_root), key="p_pdf", disabled=True)
    st.text_input("BACKUP_ROOT", str(PATHS.backup_root), key="p_bak", disabled=True)
    if hasattr(PATHS, "data_root"):
        st.text_input("DATA_ROOT", str(PATHS.data_root), key="p_data", disabled=True)

    # ---- デモ用サンプル（カテゴリ→質問選択、ランダム挿入、即送信）----
    st.divider()
    st.subheader("🧪 デモ用サンプル質問")
    cat = st.selectbox("カテゴリを選択", ["（未選択）"] + list(SAMPLES.keys()))
    sample = ""
    if cat != "（未選択）":
        sample = st.selectbox("サンプル質問を選択", ["（未選択）"] + SAMPLES[cat])
    else:
        st.caption("カテゴリを選ぶか、下のランダム挿入を使えます。")

    cols_demo = st.columns(2)
    with cols_demo[0]:
        st.button(
            "⬇️ この質問を入力欄へセット",
            use_container_width=True,
            disabled=(sample in ("", "（未選択）")),
            on_click=lambda: _set_question(sample),
        )
    with cols_demo[1]:
        st.button(
            "🎲 ランダム挿入",
            use_container_width=True,
            on_click=lambda: _set_question(random.choice(ALL_SAMPLES)),
        )
    send_now = st.button(
        "🚀 サンプルで即送信",
        use_container_width=True,
        disabled=(st.session_state.q.strip() == ""),
    )

# ---- 入力欄（セッション連携）----
q = st.text_area(
    "質問を入力",
    value=st.session_state.q,
    placeholder="この社内ボットに質問してください…",
    height=100,
)
if q != st.session_state.q:
    st.session_state.q = q

# ---- 送信判定（通常送信 or サンプル即送信）----
go_click = st.button("送信", type="primary")
go = go_click or bool(locals().get("send_now"))

# =========================
# 実行（検索 → 生成 → コスト表示 → 参照表示）
# =========================
if go and st.session_state.q.strip():
    api_key = _get_openai_api_key()

    try:
        # ---------- 検索フェーズ ----------
        # - ベクトルストア（OpenAI配下）のシャードを列挙
        # - Top-K を各シャードで取り、スコアでグローバル上位Kにマージ
        with st.spinner("検索中…"):
            vs_backend_dir = VS_ROOT / "openai"
            if not vs_backend_dir.exists():
                st.warning(f"ベクトルが見つかりません（{vs_backend_dir}）。先に **03 ベクトル化** を OpenAI で実行してください。")
                st.stop()

            # ▼ 対象シャードの決定（未指定なら全シャード）
            shard_dirs_all = _list_shard_dirs_openai()
            selected = [vs_backend_dir / s for s in target_shards] if target_shards \
                       else [vs_backend_dir / p.name for p in shard_dirs_all]
            shard_dirs = [p for p in selected if p.is_dir() and (p / "vectors.npy").exists()]
            if not shard_dirs:
                st.warning("検索可能なシャードがありません。対象シャードのベクトル化を先に実行してください。")
                st.stop()

            # ▼ クエリ内インライン指定 [[files: a.pdf, b.pdf]]
            inline = re.search(r"\[\[\s*files\s*:\s*([^\]]+)\]\]", st.session_state.q, flags=re.IGNORECASE)
            inline_files = set()
            if inline:
                inline_files = {s.strip() for s in inline.group(1).split(",") if s.strip()}

            # ▼ 参照ファイルのホワイトリスト（UI指定 + インライン指定の和）
            effective_whitelist = {_norm_path(x) for x in (set(file_whitelist) | set(inline_files))}

            # ▼ 実際に埋め込む質問テキスト（インライン指定は削除して正規化）
            clean_q = re.sub(r"\[\[\s*files\s*:[^\]]+\]\]", "", st.session_state.q, flags=re.IGNORECASE).strip()
            question = normalize_ja_text(clean_q)

            # ▼ OpenAI 埋め込み（単一クエリ）
            estore = EmbeddingStore(backend="openai")
            emb_tokens = _count_tokens(question, model_hint="text-embedding-3-large")
            qv = estore.embed([question]).astype("float32")  # shape=(1, d)

            # ▼ 各シャードのTop-K結果を最大ヒープでマージ（スコア降順）
            K = int(top_k)
            heap_: List[Tuple[float, int, int, Dict[str, Any]]] = []
            tiebreak = count()  # スコア一致時のタイブレーク

            for shp in shard_dirs:
                try:
                    vdb = NumpyVectorDB(shp)
                    # return_="similarity" を指定し (row_idx, score, meta) か (score, meta) を得る
                    hits = vdb.search(qv, top_k=K, return_="similarity")
                    for h in hits:
                        if isinstance(h, tuple) and len(h) == 3:
                            row_idx, score, meta = h
                        elif isinstance(h, tuple) and len(h) == 2:
                            score, meta = h
                            row_idx = -1
                        else:
                            continue

                        md = dict(meta or {})
                        md["shard_id"] = shp.name

                        # ▼ ファイルホワイトリストがあればフィルタ
                        if effective_whitelist:
                            if _norm_path(str(md.get("file", ""))) not in effective_whitelist:
                                continue

                        sc = float(score)
                        tb = next(tiebreak)

                        # ▼ 上位Kを維持する最小ヒープ
                        if len(heap_) < K:
                            heapq.heappush(heap_, (sc, tb, row_idx, md))
                        else:
                            if sc > heap_[0][0]:
                                heapq.heapreplace(heap_, (sc, tb, row_idx, md))
                except Exception as e:
                    # シャード単位の検索エラーは全体停止せず警告のみに留める
                    st.warning(f"シャード {shp.name} の検索でエラー: {e}")

            # ▼ スコア降順に並べ替えて上位ヒットを得る
            raw_hits = [(rid, sc, md) for (sc, _tb, rid, md) in sorted(heap_, key=lambda x: x[0], reverse=True)]

        # ヒットなし時はユーザーに通知して停止
        if not raw_hits:
            if effective_whitelist:
                st.warning("指定された参照ファイル内で該当コンテキストが見つかりませんでした。年/ファイル名（例: 2025/foo.pdf）をご確認ください。")
            else:
                st.warning("該当コンテキストが見つかりませんでした。チャンクサイズや Top-K を調整して再試行してください。")
            st.stop()

        # ---------- 回答生成フェーズ ----------
        # - プロンプト構築（S番号付きの文脈を付与）
        # - lib/gpt_responder 経由で生成（ストリーミング or 一括）
        # - ストリーミング不可エラー時は自動フォールバック
        chat_prompt_tokens = 0
        chat_completion_tokens = 0
        answer = None

        use_answer_backend = answer_backend
        if use_answer_backend == "OpenAI" and not api_key:
            use_answer_backend = "Retrieve-only"

        if use_answer_backend == "OpenAI":
            # プロンプト用のラベル付き文脈を生成
            labeled_contexts = [
                f"[S{i}] {meta.get('text','')}\n[meta: {_fmt_source(meta)} / score={float(score):.3f}]"
                for i, (_rid, score, meta) in enumerate(raw_hits, 1)
            ]
            prompt = build_prompt(
                question,
                labeled_contexts,
                sys_inst=sys_inst,
                style_hint=detail,
                cite=cite,
                strict=False,
            )

            # ★ 共通ラッパ初期化（APIキーは secrets / env を優先）
            responder = GPTResponder(api_key=api_key)

            if display_mode == "逐次表示（ストリーミング）":
                st.subheader("🧠 回答（逐次表示）")

                # 逐次出力を Streamlit にそのまま流す（内部でフォールバックも処理）
                st.write_stream(
                    responder.stream(
                        model=chat_model,
                        system_instruction=sys_inst,
                        user_content=prompt,
                        max_output_tokens=int(max_tokens),
                        on_error_text="Responses stream error.",
                    )
                )
                # ストリーム完了後の最終値を取得（フォールバックでも設定済み）
                answer = responder.final_text
                chat_prompt_tokens = responder.usage.input_tokens
                chat_completion_tokens = responder.usage.output_tokens

            else:
                # 一括生成（非ストリーム）
                with st.spinner("回答生成中…"):
                    result: CompletionResult = responder.complete(
                        model=chat_model,
                        system_instruction=sys_inst,
                        user_content=prompt,
                        max_output_tokens=int(max_tokens),
                    )
                answer = result.text
                chat_prompt_tokens = result.usage.input_tokens
                chat_completion_tokens = result.usage.output_tokens

                st.subheader("🧠 回答")
                st.write(answer)

        else:
            # 回答生成を行わない（コンテキスト確認用など）
            st.subheader("🧩 取得のみ（要約なし）")
            st.info("Retrieve-only モードです。下の参照コンテキストを参照してください。")

        # ---------- コスト表示フェーズ ----------
        # - 埋め込みの概算コスト + 回答生成の概算コスト（USD→JPY）を算出
        with st.container():
            emb_cost_usd = estimate_embedding_cost(
                "text-embedding-3-large",
                _count_tokens(st.session_state.q, "text-embedding-3-large"),
            )["usd"]
            chat_cost_usd = 0.0
            if use_answer_backend == "OpenAI":
                chat_cost_usd = estimate_chat_cost(
                    chat_model,
                    ChatUsage(input_tokens=chat_prompt_tokens, output_tokens=chat_completion_tokens),
                )["usd"]

            total_usd = emb_cost_usd + chat_cost_usd
            total_jpy = usd_to_jpy(total_usd, rate=DEFAULT_USDJPY)

            # メトリクス：合計JPY / 内訳USD / 単価（USD/1K）
            st.markdown("### 💴 使用料の概算（lib/costs による集計）")
            cols = st.columns(3)
            with cols[0]:
                st.metric("合計 (JPY)", f"{total_jpy:,.2f} 円")
                st.caption(f"為替 {DEFAULT_USDJPY:.2f} JPY/USD")
            with cols[1]:
                st.write("**内訳 (USD)**")
                st.write(f"- Embedding: `${emb_cost_usd:.6f}`")
                if use_answer_backend == "OpenAI":
                    st.write(
                        f"- Chat 合計: `${chat_cost_usd:.6f}` "
                        f"(in={chat_prompt_tokens} tok / out={chat_completion_tokens} tok)"
                    )
                st.write(f"- 合計: `${total_usd:.6f}`")
            with cols[2]:
                emb_price_per_1k = float(EMBEDDING_PRICES_USD.get("text-embedding-3-large", 0.0)) / 1000.0
                st.write("**単価 (USD / 1K tok)**")
                st.write(f"- Embedding: `${emb_price_per_1k:.5f}`（text-embedding-3-large）")
                st.write(f"- Chat 入力: `${MODEL_PRICES_PER_1K.get(chat_model,{}).get('in',0.0):.5f}`（{chat_model}）")
                st.write(f"- Chat 出力: `${MODEL_PRICES_PER_1K.get(chat_model,{}).get('out',0.0):.5f}`（{chat_model}）")

        # ---------- 参照コンテキスト（上位ヒット） ----------
        with st.expander("🔍 参照コンテキスト（上位ヒット）", expanded=False):
            for i, (_rid, score, meta) in enumerate(raw_hits, 1):
                txt = str(meta.get("text", "") or "")
                src_label = _fmt_source(meta)
                snippet = (txt[:1000] + "…") if len(txt) > 1000 else txt
                st.markdown(f"**[S{i}] score={float(score):.3f}**  `{src_label}`\n\n{snippet}")

    except Exception as e:
        # 検索/生成を含む全体の例外ハンドリング（UIに可視化）
        st.error(f"検索/生成中にエラー: {e}")
else:
    # 初期状態のガイド
    st.info("質問を入力して『送信』を押してください。サイドバーでシャードや回答設定、参照ファイルを調整できます。")
