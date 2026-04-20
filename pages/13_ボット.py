# -*- coding: utf-8 -*-
# ============================================================
# bot_app/pages/13_ボット.py
# ============================================================
# 📝 社内ボット（RAG）
#
# ✅ テキストAI最新テンプレ準拠（ページ側）
# - ログイン正本：page_session_heartbeat（require_login は使わない）
# - AI実行：with busy_run（ai_runs.db を正本として記録）
# - tokens/cost：返ってきた範囲のみ br.set_usage / br.set_cost（推計しない）
# - 直近ラン表示：render_run_summary_compact（go ブロック外 / rerun耐性）
# - st.form 不使用 / use_container_width 不使用
# - st.button()/st.download_button() に width 引数は使わない
#
# ✅ 既存機能は維持
# - 利用者履歴（SQLite）   : Storages/<sub>/bot_app/ 配下（現状どおり）
# - 管理者監査ログ（JSONL）: JsonlLogger（monthly rotate）
# - 出典拡張 expander / 参照コンテキスト expander
# - Word 保存（.docx）
# - Inbox 保存（質問+回答 .txt）※ go ブロック外（rerun耐性）
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Set, Optional, Tuple
import sys
import re
import inspect
import json
import datetime as dt

import streamlit as st

# ============================================================
# ページ設定（最初に1回だけ）
# ============================================================
st.set_page_config(
    page_title="Bot / Chat Bot",
    page_icon="💬",
    layout="wide",
)

# ============================================================
# sys.path 調整（設計上の確定事項）
# ============================================================
_THIS = Path(__file__).resolve()
APP_DIR = _THIS.parents[1]
PROJ_DIR = _THIS.parents[2]
MONO_ROOT = _THIS.parents[3]

for p in (MONO_ROOT, PROJ_DIR, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PROJECTS_ROOT = MONO_ROOT
APP_NAME = "bot_app"
PAGE_NAME = _THIS.stem

JST = dt.timezone(dt.timedelta(hours=9), name="Asia/Tokyo")

# ============================================================
# imports（既存）
# ============================================================
from config.sample_questions import SAMPLES2
from config.config import has_gemini_api_key

from lib.bot.pipeline import run_bot_query, BotAnswerView
from lib.bot_utils import (
    parse_years,
    parse_pnos,
    norm_pno_forms,
    fmt_source,
)

from lib.bot.docx_export import (
    docx_available,
    make_bot_answer_docx_bytes,
    make_default_docx_filename,
)

from lib.bot.explanation import render_bot_usage_expander

from common_lib.logs.jsonl_logger import JsonlLogger, sha256_short

from lib.logs.sqlite_logger import (
    init_bot_logs_db,
    insert_bot_log_row,
    preview_text,
)

from common_lib.storage.external_ssd_root import resolve_storage_subdir_root

from common_lib.inbox.inbox_ops.ingest import ingest_to_inbox
from common_lib.inbox.inbox_common.types import (
    IngestRequest,
    InboxNotAvailable,
    QuotaExceeded,
    IngestFailed,
)

from common_lib.ui.banner_lines import render_banner_line_by_key

# ============================================================
# imports（テンプレ準拠：ログイン/Busy/UI）
# ============================================================
from common_lib.sessions.page_entry import page_session_heartbeat
from common_lib.busy import busy_run
from common_lib.ui import render_run_summary_compact

# ============================================================
# Models / Constants（既存のUI仕様を維持）
# ============================================================
OPENAI_MODELS = ["gpt-5-mini", "gpt-5-nano"]
GEMINI_MODELS = ["gemini-2.0-flash"]
MAX_OUTPUT_TOKENS = 12_000

INCLUDE_FULL_PROMPT_IN_LOG = True
INCLUDE_FULL_ANSWER_IN_LOG = True

# ============================================================
# Streamlit state keys（ページ専用 / 事故防止）
# ============================================================
K_Q = f"{PAGE_NAME}__q"

K_LAST_ANSWER = f"{PAGE_NAME}__last_answer"
K_INBOX_SAVE_MSG = f"{PAGE_NAME}__inbox_save_msg"

# 直近ラン表示（テンプレ準拠）
K_LAST_RUN_ID = f"{PAGE_NAME}__last_run_id"
K_LAST_MODEL = f"{PAGE_NAME}__last_model"
K_LAST_PROVIDER = f"{PAGE_NAME}__last_provider"
K_LAST_IN_TOK = f"{PAGE_NAME}__last_in_tok"
K_LAST_OUT_TOK = f"{PAGE_NAME}__last_out_tok"
K_LAST_COST_OBJ = f"{PAGE_NAME}__last_cost_obj"
K_LAST_NOTE = f"{PAGE_NAME}__last_note"

st.session_state.setdefault(K_Q, "")

st.session_state.setdefault(K_LAST_ANSWER, None)
st.session_state.setdefault(K_INBOX_SAVE_MSG, None)

st.session_state.setdefault(K_LAST_RUN_ID, None)
st.session_state.setdefault(K_LAST_MODEL, "")
st.session_state.setdefault(K_LAST_PROVIDER, "")
st.session_state.setdefault(K_LAST_IN_TOK, None)
st.session_state.setdefault(K_LAST_OUT_TOK, None)
st.session_state.setdefault(K_LAST_COST_OBJ, None)
st.session_state.setdefault(K_LAST_NOTE, "")

# ============================================================
# helpers（互換 / 表示 / 保存）
# ============================================================
def _get_provider_label_from_model(model: str) -> str:
    """
    provider の分類（推計ではなくラベル付け）
    - gemini* -> google
    - それ以外 -> openai
    """
    return "google" if str(model).strip().lower().startswith("gemini") else "openai"

def _run_bot_query_compat(
    *,
    question: str,
    chat_model: str,
    detail: str,
    max_tokens: int,
    top_k: int,
    years_sel: Set[int],
    pnos_sel_norm: Set[str],
    system_instruction: str,
    vectorspace: str,
    strict: bool,
) -> BotAnswerView:
    """
    互換目的の例外（テンプレ要件ではない）：
    pipeline 側の run_bot_query が引数拡張されても落ちないように、
    signature を見て渡せるものだけ渡す。
    """
    try:
        sig = inspect.signature(run_bot_query)
        params = sig.parameters

        kwargs = dict(
            question=question,
            chat_model=chat_model,
            detail=detail,
            max_tokens=max_tokens,
            top_k=top_k,
            years_sel=years_sel,
            pnos_sel_norm=pnos_sel_norm,
            system_instruction=system_instruction,
        )

        if "vectorspace" in params:
            kwargs["vectorspace"] = vectorspace
        if "strict" in params:
            kwargs["strict"] = bool(strict)

        return run_bot_query(**kwargs)  # type: ignore[arg-type]

    except Exception:
        # 互換 fallback（古いI/F向け）
        return run_bot_query(
            question=question,
            chat_model=chat_model,
            detail=detail,
            max_tokens=max_tokens,
            top_k=top_k,
            years_sel=years_sel,
            pnos_sel_norm=pnos_sel_norm,
            system_instruction=system_instruction,
        )


def _set_q(x: str) -> None:
    st.session_state[K_Q] = x or ""


def build_bot_qa_txt(
    *,
    created_at_iso: str,
    app_name: str,
    page_name: str,
    user: str,
    vectorspace: str,
    chat_model: str,
    detail_label: str,
    detail: str,
    top_k: int,
    max_tokens: int,
    prompt_text: str,
    answer_text: str,
    prompt_hash: str,
    answer_hash: str,
    cost_usd: float | None,
    cost_jpy: float | None,
    embedding_tokens: int | None,
    chat_input_tokens: int | None,
    chat_output_tokens: int | None,
) -> str:
    # AIに渡しやすい「固定キー:値」+ セクション分離
    # ※改行・順序は崩さないほうが後工程で安定
    lines: list[str] = []
    lines.append("=== BOT_QA ===")
    lines.append(f"created_at: {created_at_iso}")
    lines.append(f"app: {app_name}")
    lines.append(f"page: {page_name}")
    lines.append(f"user: {user}")
    lines.append(f"vectorspace: {vectorspace}")
    lines.append(f"model: {chat_model}")
    lines.append(f"detail_label: {detail_label}")
    lines.append(f"detail: {detail}")
    lines.append(f"top_k: {int(top_k)}")
    lines.append(f"max_tokens: {int(max_tokens)}")
    lines.append(f"prompt_hash: {prompt_hash}")
    lines.append(f"answer_hash: {answer_hash}")
    lines.append(f"embedding_tokens: {embedding_tokens}")
    lines.append(f"chat_input_tokens: {chat_input_tokens}")
    lines.append(f"chat_output_tokens: {chat_output_tokens}")
    lines.append(f"cost_usd: {cost_usd}")
    lines.append(f"cost_jpy: {cost_jpy}")
    lines.append("")

    lines.append("=== QUESTION ===")
    lines.append(prompt_text.rstrip())
    lines.append("")

    lines.append("=== ANSWER ===")
    lines.append(answer_text.rstrip())
    lines.append("")

    return "\n".join(lines) + "\n"


# ============================================================
# （テンプレ準拠）直近ラン表示用 cost オブジェクト（usd/jpy 属性だけ持つ）
# ============================================================
class _CostView:
    def __init__(self, usd: Optional[float], jpy: Optional[float]) -> None:
        self.usd = usd
        self.jpy = jpy


# ============================================================
# banner
# ============================================================
render_banner_line_by_key("cyan_clean")

# ============================================================
# ログイン（正本）：page_session_heartbeat
# ============================================================
sub = page_session_heartbeat(
    st,
    PROJECTS_ROOT,
    app_name=APP_NAME,
    page_name=PAGE_NAME,
)
user_sub = str(sub)
owner_sub = str(sub)

# ============================================================
# タイトル部
# ============================================================
left, right = st.columns([2, 1])
with left:
    st.title("📝 社内ボット")
with right:
    st.success(f"✅ ログイン中: **{user_sub}**")

# ============================================================
# 説明文（既存のまま）
# ============================================================
st.caption(
    "社内ボットはRAGを用いたアプリです．"
    "RAG（retrieval augmented generation）は、あらかじめ用意した資料やデータベースから関連情報を検索（retrieval）し、その結果だけを文脈として文章生成（generation）を行う手法です。"
    "これにより、AIが資料に含まれない情報を勝手に補完することを防ぎ、根拠が明確で再現性の高い回答が可能になります。"
    "社内文書検索やFAQなど、「どの資料に基づいて答えたか」が重要な場面で特に有効です。"
    "この仕組みにより、資料に含まれていない内容は構造的に参照できず、原則として回答されません。"
)

st.caption(
    "検索DBはreport(openai）を用いてください．現在2019年と2020年の報告書のデータがデータベース化されています．"
    "データを整理しながら，データベースへの登録を進めている状態です．最終的には全ての報告書が登録される予定です．"
)

st.caption(
    "「検索件数（Top-K)」は最初にデータベースから取り出すデータの数です．デフォルトで6（一般的な推奨値）が設定されています．"
    "検索結果が全く出ない時などは，この数を増やしてみて，再度送信を行ってみてください．"
)

st.caption(
    "現状では，回答の確率と日本語としての自然さを向上させるために，デフォルトで「補足あり（一般知識からも補足）」になっています．"
    "クリーンな報告書のデータベースが整備された後は，デフォルトで「厳格（補足なし）」に変更する予定です．")

st.caption("検索結果をInboxへ送り，保存することができます．")

# ---- 前回のInbox保存結果（rerun後も見えるように）----
if st.session_state.get(K_INBOX_SAVE_MSG):
    msg, level = st.session_state[K_INBOX_SAVE_MSG]
    if level == "success":
        st.success(msg)
    else:
        st.error(msg)
    # 1回表示したら消す
    st.session_state[K_INBOX_SAVE_MSG] = None

# ============================================================
# Storages root（settings.toml は使わない）
# ============================================================
STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)

# ============================================================
# (A) 利用者履歴（SQLite）：Storages/<sub>/bot_app/
# ============================================================
base_dir = STORAGE_ROOT / owner_sub / APP_NAME
(base_dir / "logs").mkdir(parents=True, exist_ok=True)

# SQLite 初期化（利用者専用）
_SQLITE_DB_PATH = init_bot_logs_db(base_dir)

# ============================================================
# (B) 管理者監査ログ（JSONL）：monthly rotate
# ============================================================
logger = JsonlLogger(
    projects_root=PROJECTS_ROOT,
    app_name=APP_NAME,
    page_name=PAGE_NAME,
    log_name=APP_NAME,   # ファイル名ベース
    rotate="monthly",
)

# ============================================================
# --- 使い方（expander）---
# ============================================================
render_bot_usage_expander(expanded=False)
st.divider()

# ============================================================
# Sidebar（本体UI）
# ============================================================
with st.sidebar:
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

    model_options = list(OPENAI_MODELS)
    if has_gemini_api_key():
        model_options += GEMINI_MODELS

    chat_model = st.radio(
        "モデル",
        model_options,
        index=0,
        help="Gemini は API キー設定時のみ表示されます。",
    )

    top_k = st.slider("検索件数（Top-K）", 1, 12, 6, 1)

    detail_label = st.selectbox("詳しさ", ["簡潔", "標準", "詳細", "超詳細"], index=2)
    detail = {
        "簡潔": "concise",
        "標準": "standard",
        "詳細": "detailed",
        "超詳細": "very_detailed",
    }[detail_label]

    # 回答モード（strict）
    st.divider()
    st.subheader("回答モード（strict）")

    strict_label = st.radio(
        "モード",
        options=[
            "厳格（資料外は参照しない）",
            "補足あり（一般知識からも補足）",
        ],
        index=1,
        help="厳格: Retrieved Contexts のみで回答。補足あり: 不足分のみ一般知識で補足（補足は明示）。",
    )
    strict_mode = (strict_label == "厳格（資料外は参照しない）")


    # 既存仕様：固定
    system_instruction = "あなたは優秀な社内のアシスタントです."

    st.divider()
    st.subheader("year / pno フィルタ")

    years_input = st.text_input("year（任意）", value="")
    pnos_input = st.text_input("pno（任意）", value="")

    years_sel: Set[int] = parse_years(years_input)
    pnos_raw: Set[str] = parse_pnos(pnos_input)
    pnos_sel_norm: Set[str] = set()
    for p in pnos_raw:
        pnos_sel_norm |= norm_pno_forms(p)

    st.caption(
        f"year: {sorted(years_sel) or '未指定'} / "
        f"pno: {sorted(pnos_sel_norm) or '未指定'}"
    )

    st.divider()
    st.subheader("🧪 サンプル質問")

    cat = st.selectbox("カテゴリ", ["（未選択）"] + list(SAMPLES2.keys()))
    samples = ["（未選択）"] if cat == "（未選択）" else ["（未選択）"] + SAMPLES2.get(cat, [])
    sample = st.selectbox("質問例", samples)

    st.button(
        "⬇️ 入力欄にセット",
        disabled=(sample == "（未選択）"),
        on_click=lambda: _set_q(sample),
    )

# ============================================================
# Main（質問入力）
# ============================================================
q = st.text_area(
    "質問を入力",
    value=st.session_state.get(K_Q, ""),
    height=100,
    placeholder="社内ボットに質問してください…",
)

if q != st.session_state.get(K_Q, ""):
    st.session_state[K_Q] = q

go = st.button("送信", type="primary")

# ============================================================
# 実行
# ============================================================
if go and (st.session_state.get(K_Q) or "").strip():
    prompt_text = (st.session_state.get(K_Q) or "").strip()

    # SQLite には vectorspace を埋め込んで残す（スキーマ変更なし）
    page_for_sqlite = f"{PAGE_NAME}__{vectorspace}"

    provider_label = _get_provider_label_from_model(chat_model)

    # ------------------------------------------------------------
    # ask log (jsonl：管理者用)
    # ------------------------------------------------------------
    logger.append(
        {
            "user": user_sub,
            "action": "ask",
            "vectorspace": vectorspace,
            "strict": bool(strict_mode),
            "chat_model": chat_model,
            "detail_label": detail_label,
            "detail": detail,
            "top_k": int(top_k),
            "max_tokens": MAX_OUTPUT_TOKENS,
            "prompt_hash": sha256_short(prompt_text),
            **({"prompt": prompt_text} if INCLUDE_FULL_PROMPT_IN_LOG else {}),
        }
    )

    # ------------------------------------------------------------
    # ask log (sqlite：利用者履歴)
    # ------------------------------------------------------------
    insert_bot_log_row(
        _SQLITE_DB_PATH,
        app=APP_NAME,  # ★固定名に統一
        page=page_for_sqlite,
        user=user_sub,
        action="ask",
        model=chat_model,
        detail=detail,
        embedding_tokens=None,
        input_tokens=None,
        output_tokens=None,
        cost_usd=None,
        cost_jpy=None,
        prompt_hash=sha256_short(prompt_text),
        answer_hash=None,
        prompt=prompt_text if INCLUDE_FULL_PROMPT_IN_LOG else None,
        answer=None,
    )

    # ------------------------------------------------------------
    # ✅（テンプレ準拠）AI実行：busy_run（ai_runs.db 正本）
    # ------------------------------------------------------------
    try:
        answer_view: Optional[BotAnswerView] = None
        run_id: str = ""
        in_tok: Optional[int] = None
        out_tok: Optional[int] = None
        usd: Optional[float] = None
        jpy: Optional[float] = None
        note: str = ""

        with busy_run(
            projects_root=PROJECTS_ROOT,
            user_sub=str(user_sub),
            app_name=str(APP_NAME),
            page_name=str(PAGE_NAME),
            task_type="rag",
            provider=str(provider_label),
            model=str(chat_model),
            meta={
                "feature": "bot_rag",
                "vectorspace": str(vectorspace),
                "strict": bool(strict_mode),
                "detail": str(detail),
                "detail_label": str(detail_label),
                "top_k": int(top_k),
                "max_tokens": int(MAX_OUTPUT_TOKENS),
                "filters": {
                    "years_sel": sorted(list(years_sel)) if years_sel else [],
                    "pnos_sel_norm": sorted(list(pnos_sel_norm)) if pnos_sel_norm else [],
                },
                "prompt_chars": len(prompt_text),
            },
        ) as br:
            run_id = br.run_id  # 直近ラン表示に使う

            # ------------------------------------------------------------
            # spinner：RAG検索＋回答生成の処理中表示
            # ------------------------------------------------------------
            with st.spinner("検索中…（RAG）"):
                # --- pipeline ---
                answer_view = _run_bot_query_compat(
                    question=prompt_text,
                    chat_model=chat_model,
                    detail=detail,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    top_k=int(top_k),
                    years_sel=years_sel,
                    pnos_sel_norm=pnos_sel_norm,
                    system_instruction=system_instruction,
                    vectorspace=vectorspace,
                    strict=bool(strict_mode),
                )

            # ---- tokens（推計しない：取れた範囲だけ）----
            in_tok = getattr(answer_view, "chat_input_tokens", None)
            out_tok = getattr(answer_view, "chat_output_tokens", None)
            if isinstance(in_tok, int) and isinstance(out_tok, int):
                br.set_usage(int(in_tok), int(out_tok))

            # ---- cost（推計しない：取れた範囲だけ）----
            usd = getattr(answer_view, "cost_usd", None)
            jpy = getattr(answer_view, "cost_jpy", None)
            if isinstance(usd, (int, float)) and isinstance(jpy, (int, float)):
                br.set_cost(float(usd), float(jpy))

            note = "gemini_estimated" if bool(getattr(answer_view, "used_gemini", False)) else ""
            br.add_finish_meta(note=note)


            

        if answer_view is None:
            raise RuntimeError("BotAnswerView が取得できませんでした。")

        # --------------------------------------------------------
        # 直近ラン（テンプレ準拠：共通UI）
        # --------------------------------------------------------
        st.session_state[K_LAST_RUN_ID] = run_id
        st.session_state[K_LAST_MODEL] = str(chat_model)
        st.session_state[K_LAST_PROVIDER] = str(provider_label)
        st.session_state[K_LAST_IN_TOK] = in_tok if isinstance(in_tok, int) else None
        st.session_state[K_LAST_OUT_TOK] = out_tok if isinstance(out_tok, int) else None
        st.session_state[K_LAST_COST_OBJ] = _CostView(
            float(usd) if isinstance(usd, (int, float)) else None,
            float(jpy) if isinstance(jpy, (int, float)) else None,
        )
        st.session_state[K_LAST_NOTE] = note

    except Exception as e:
        st.error(f"AI 呼び出しでエラー: {e}")
        st.stop()

    # ------------------------------------------------------------
    # --- この回答を「後から保存」できるように保持（Streamlit rerun対策）---
    # ------------------------------------------------------------
    st.session_state[K_LAST_ANSWER] = {
        "prompt_text": prompt_text,
        "answer_text": answer_view.answer_text,
        "vectorspace": vectorspace,
        "strict": bool(strict_mode),
        "chat_model": chat_model,
        "detail_label": detail_label,
        "detail": detail,
        "top_k": int(top_k),
        "max_tokens": MAX_OUTPUT_TOKENS,
        "embedding_tokens": getattr(answer_view, "embedding_tokens", None),
        "chat_input_tokens": getattr(answer_view, "chat_input_tokens", None),
        "chat_output_tokens": getattr(answer_view, "chat_output_tokens", None),
        "cost_usd": getattr(answer_view, "cost_usd", None),
        "cost_jpy": getattr(answer_view, "cost_jpy", None),
        "used_gemini": bool(getattr(answer_view, "used_gemini", False)),
        "debug_system": getattr(answer_view, "debug_system", "") if answer_view else "",
        "debug_user": getattr(answer_view, "debug_user", "") if answer_view else "",
    }

    # ------------------------------------------------------------
    # --- 回答 ---
    # ------------------------------------------------------------
    st.write(answer_view.answer_text)

    # ------------------------------------------------------------
    # --- 出典拡張 ---
    # ------------------------------------------------------------
    citations = re.findall(r"\[S[^\]]+\]", answer_view.answer_text)
    if citations:
        uniq: list[str] = []
        for c in citations:
            if c not in uniq:
                uniq.append(c)
        with st.expander("📝 出典拡張済み最終テキスト", expanded=False):
            st.text("\n".join(uniq))

    # ------------------------------------------------------------
    # --- 参照コンテキスト ---
    # ------------------------------------------------------------
    with st.expander("🔍 参照コンテキスト（上位ヒット）", expanded=False):
        for i, (_rid, score, meta) in enumerate(answer_view.raw_hits, 1):
            snippet = (meta.get("text", "") or "")[:1000]
            st.markdown(
                f"**[S{i}] score={score:.3f}**  `{fmt_source(meta)}`\n\n{snippet}"
            )

    # ------------------------------------------------------------
    # --- Word ---
    # ------------------------------------------------------------
    if docx_available():
        st.download_button(
            "⬇️ プロンプト＋回答を Word で保存 (.docx)",
            data=make_bot_answer_docx_bytes(
                prompt_text=prompt_text,
                answer_text=answer_view.answer_text,
                user=user_sub,
                chat_model=chat_model,
                detail_label=detail_label,
                detail=detail,
                max_tokens=MAX_OUTPUT_TOKENS,
                top_k=int(top_k),
                years_sel=years_sel,
                pnos_sel_norm=pnos_sel_norm,
                shards=[vectorspace],
            ),
            file_name=make_default_docx_filename(prefix="bot_answer"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    # ------------------------------------------------------------
    # --- answer log (jsonl：管理者用)
    # ------------------------------------------------------------
    logger.append(
        {
            "user": user_sub,
            "action": "answer",
            "vectorspace": vectorspace,
            "chat_model": chat_model,
            "detail": detail,
            "embedding_tokens": getattr(answer_view, "embedding_tokens", None),
            "chat_input_tokens": getattr(answer_view, "chat_input_tokens", None),
            "chat_output_tokens": getattr(answer_view, "chat_output_tokens", None),
            "cost_usd": getattr(answer_view, "cost_usd", None),
            "cost_jpy": getattr(answer_view, "cost_jpy", None),
            "prompt_hash": sha256_short(prompt_text),
            "answer_hash": sha256_short(answer_view.answer_text),
            "answer_preview": preview_text(answer_view.answer_text, 20),
            **({"answer": answer_view.answer_text} if INCLUDE_FULL_ANSWER_IN_LOG else {}),
        }
    )

    # ------------------------------------------------------------
    # --- answer log (sqlite：利用者履歴)
    # ------------------------------------------------------------
    insert_bot_log_row(
        _SQLITE_DB_PATH,
        app=APP_NAME,
        page=page_for_sqlite,
        user=user_sub,
        action="answer",
        model=chat_model,
        detail=detail,
        embedding_tokens=getattr(answer_view, "embedding_tokens", None),
        input_tokens=getattr(answer_view, "chat_input_tokens", None),
        output_tokens=getattr(answer_view, "chat_output_tokens", None),
        cost_usd=getattr(answer_view, "cost_usd", None),
        cost_jpy=getattr(answer_view, "cost_jpy", None),
        prompt_hash=sha256_short(prompt_text),
        answer_hash=sha256_short(answer_view.answer_text),
        prompt=prompt_text if INCLUDE_FULL_PROMPT_IN_LOG else None,
        answer=answer_view.answer_text if INCLUDE_FULL_ANSWER_IN_LOG else None,
    )

else:
    st.info("質問を入力して『送信』を押してください。")

# ============================================================
# ✅（テンプレ準拠）直近ラン（コンパクト表示：共通UI）
# ============================================================
if st.session_state.get(K_LAST_RUN_ID):
    render_run_summary_compact(
        projects_root=PROJECTS_ROOT,
        run_id=str(st.session_state.get(K_LAST_RUN_ID)),
        model=str(st.session_state.get(K_LAST_MODEL) or "—"),
        in_tokens=st.session_state.get(K_LAST_IN_TOK),
        out_tokens=st.session_state.get(K_LAST_OUT_TOK),
        cost=st.session_state.get(K_LAST_COST_OBJ),
        note=str(st.session_state.get(K_LAST_NOTE) or ""),
        show_divider=True,
    )

# ============================================================
# 📥 Inboxへ保存（.txt：質問+回答）— go ブロック外（rerunで消えない）
# ============================================================
last = st.session_state.get(K_LAST_ANSWER)

if last and (last.get("prompt_text") or "").strip() and (last.get("answer_text") or "").strip():
    st.divider()
    st.subheader("💾 保存（直近の回答）")
    st.caption("直近に生成した回答を、AI入力向け .txt として Inbox に保存します。タグ：bot回答 + 日付（JST）")

    save_to_inbox2 = st.button("📥 Inboxへ保存（質問+回答 .txt）", key=f"{PAGE_NAME}__save_inbox_txt_btn")

    if save_to_inbox2:
        try:
            now = dt.datetime.now(JST)
            date_tag = now.strftime("%Y-%m-%d")           # タグ用
            ts_compact = now.strftime("%Y-%m-%d_%H%M")    # ファイル名用

            prompt_text2 = last["prompt_text"]
            answer_text2 = last["answer_text"]

            prompt_hash = sha256_short(prompt_text2)
            answer_hash = sha256_short(answer_text2)

            txt = build_bot_qa_txt(
                created_at_iso=now.isoformat(),
                app_name=APP_NAME,
                page_name=PAGE_NAME,
                user=user_sub,
                vectorspace=last["vectorspace"],
                chat_model=last["chat_model"],
                detail_label=last["detail_label"],
                detail=last["detail"],
                top_k=int(last["top_k"]),
                max_tokens=int(last["max_tokens"]),
                prompt_text=prompt_text2,
                answer_text=answer_text2,
                prompt_hash=prompt_hash,
                answer_hash=answer_hash,
                cost_usd=last.get("cost_usd"),
                cost_jpy=last.get("cost_jpy"),
                embedding_tokens=last.get("embedding_tokens"),
                chat_input_tokens=last.get("chat_input_tokens"),
                chat_output_tokens=last.get("chat_output_tokens"),
            )
            data = txt.encode("utf-8")

            tags = ["bot回答", date_tag]
            tags_json = json.dumps(tags, ensure_ascii=False)

            filename = f"bot_answer_{ts_compact}.txt"

            r = ingest_to_inbox(
                projects_root=PROJECTS_ROOT,
                req=IngestRequest(
                    user_sub=user_sub,
                    filename=filename,
                    data=data,
                    tags_json=tags_json,
                    origin={
                        "app": APP_NAME,
                        "page": PAGE_NAME,
                        "action": "save_to_inbox_txt",
                        "vectorspace": last["vectorspace"],
                        "chat_model": last["chat_model"],
                        "detail": last["detail"],
                        "prompt_hash": prompt_hash,
                        "answer_hash": answer_hash,
                    },
                ),
            )

            # 成功メッセージは rerun でも残るよう session に積む
            st.session_state[K_INBOX_SAVE_MSG] = (
                f"✅ Inboxへ保存しました: `{filename}` / result={str(r)}",
                "success",
            )

            # 監査ログ（任意）
            logger.append(
                {
                    "user": user_sub,
                    "action": "save_to_inbox_txt",
                    "vectorspace": last["vectorspace"],
                    "chat_model": last["chat_model"],
                    "detail": last["detail"],
                    "prompt_hash": prompt_hash,
                    "answer_hash": answer_hash,
                    "filename": filename,
                    "bytes": len(data),
                    "tags": tags,
                }
            )

            st.rerun()

        except InboxNotAvailable:
            st.session_state[K_INBOX_SAVE_MSG] = ("❌ Inbox が利用できません（InboxNotAvailable）", "error")
            st.rerun()
        except QuotaExceeded as e:
            st.session_state[K_INBOX_SAVE_MSG] = (
                "❌ Inbox 容量上限（QuotaExceeded）: "
                f"current={getattr(e,'current',None)} incoming={getattr(e,'incoming',None)} quota={getattr(e,'quota',None)}",
                "error",
            )
            st.rerun()
        except IngestFailed as e:
            st.session_state[K_INBOX_SAVE_MSG] = (f"❌ Inbox への保存に失敗しました（IngestFailed）: {e}", "error")
            st.rerun()
        except Exception as e:
            st.session_state[K_INBOX_SAVE_MSG] = (f"❌ 予期しないエラー: {e}", "error")
            st.rerun()

# ============================================================
# 🧪 デバッグ：使用したプロンプト（System / User）
# （ページ最下部）
# ============================================================
if st.session_state.get(K_LAST_ANSWER) is not None:
    # last_answer がある = 直近実行済み（最低限のガード）
    # answer_view は rerun のたびに必ず存在するとは限らないため、session から取れる情報のみを使う
    last = st.session_state.get(K_LAST_ANSWER) or {}

    # answer_view の debug_* を session に保存していない場合に備えて空にしておく
    # （pipeline 側で debug_system/debug_user を BotAnswerView に載せているなら、ここは空になりにくい）
    system_text_dbg = str(last.get("debug_system") or "")
    user_text_dbg = str(last.get("debug_user") or "")

    with st.expander("🧪 使用したプロンプト（System / User）", expanded=False):
        st.caption("※ 実際に API に渡した内容です（デバッグ用）")

        st.markdown("### System（API system_instruction）")
        st.text(system_text_dbg)

        st.markdown("### User（user_content / build_prompt の結果）")
        st.text(user_text_dbg)

