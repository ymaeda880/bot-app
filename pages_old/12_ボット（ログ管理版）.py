# pages/12_ボット（ログ管理版）.py
from __future__ import annotations

from pathlib import Path
from typing import Set
import sys
import re

import streamlit as st

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

# --- sys.path 調整 ---
_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie
from common_lib.logs.jsonl_logger import JsonlLogger, sha256_short

from lib.logs.sqlite_logger import (
    init_bot_logs_db,
    insert_bot_log_row,
    preview_text,
)

# ============================================================
# Logger
# ============================================================

_APP_DIR = Path(__file__).resolve().parents[1]
_PAGE_NAME = Path(__file__).stem

# ★ 月次ローテーションを有効化
logger = JsonlLogger(
    app_dir=_APP_DIR,
    page_name=_PAGE_NAME,
    rotate="monthly",
)

INCLUDE_FULL_PROMPT_IN_LOG = True
INCLUDE_FULL_ANSWER_IN_LOG = True

_SQLITE_DB_PATH = init_bot_logs_db(_APP_DIR)

# ============================================================
# Models / Constants
# ============================================================

OPENAI_MODELS = ["gpt-5-mini", "gpt-5-nano"]
GEMINI_MODELS = ["gemini-2.0-flash"]

# ★ 最大出力トークンは固定
MAX_OUTPUT_TOKENS = 12_000

# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="Chat Bot — Pipeline版",
    page_icon="💬",
    layout="wide",
)

col_title, col_user = st.columns([5, 2], vertical_alignment="center")
with col_title:
    st.title("💬 社内ボット")
with col_user:
    current_user, _ = get_current_user_from_session_or_cookie(st)
    if current_user:
        st.success(f"ログイン中: **{current_user}**")
    else:
        st.warning("未ログイン")

st.caption("このアプリは、操作ログ・回答ログをデータベースに記録します。")

if "q" not in st.session_state:
    st.session_state.q = ""

def _set_q(x: str) -> None:
    st.session_state.q = x or ""

# --- 使い方 ---
render_bot_usage_expander(expanded=False)

st.divider()

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("設定")

    # --- モデル選択（radio） ---
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
    detail_map = {
        "簡潔": "concise",
        "標準": "standard",
        "詳細": "detailed",
        "超詳細": "very_detailed",
    }
    detail = detail_map[detail_label]

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
        use_container_width=True,
        disabled=(sample == "（未選択）"),
        on_click=lambda: _set_q(sample),
    )

# ============================================================
# Main
# ============================================================

q = st.text_area(
    "質問を入力",
    value=st.session_state.q,
    height=100,
    placeholder="社内ボットに質問してください…",
)

if q != st.session_state.q:
    st.session_state.q = q

go = st.button("送信", type="primary")

# ============================================================
# 実行
# ============================================================

if go and st.session_state.q.strip():
    prompt_text = st.session_state.q.strip()

    # --- ask log ---
    logger.append({
        "user": current_user or "(anonymous)",
        "action": "ask",
        "chat_model": chat_model,
        "detail_label": detail_label,
        "detail": detail,
        "top_k": int(top_k),
        "max_tokens": MAX_OUTPUT_TOKENS,
        "prompt_hash": sha256_short(prompt_text),
        **({"prompt": prompt_text} if INCLUDE_FULL_PROMPT_IN_LOG else {}),
    })

    insert_bot_log_row(
        _SQLITE_DB_PATH,
        app=_APP_DIR.name,
        page=_PAGE_NAME,
        user=current_user or "(anonymous)",
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

    # --- pipeline ---
    view: BotAnswerView = run_bot_query(
        question=prompt_text,
        chat_model=chat_model,
        detail=detail,
        max_tokens=MAX_OUTPUT_TOKENS,
        top_k=int(top_k),
        years_sel=years_sel,
        pnos_sel_norm=pnos_sel_norm,
        system_instruction=system_instruction,
    )

    # --- 回答 ---
    st.subheader("🧠 回答")
    st.write(view.answer_text)

    # --- 出典拡張 ---
    citations = re.findall(r"\[S[^\]]+\]", view.answer_text)
    if citations:
        uniq = []
        for c in citations:
            if c not in uniq:
                uniq.append(c)
        with st.expander("📝 出典拡張済み最終テキスト", expanded=False):
            st.text("\n".join(uniq))

    # --- 参照コンテキスト ---
    with st.expander("🔍 参照コンテキスト（上位ヒット）", expanded=False):
        for i, (_rid, score, meta) in enumerate(view.raw_hits, 1):
            snippet = (meta.get("text", "") or "")[:1000]
            st.markdown(
                f"**[S{i}] score={score:.3f}**  `{fmt_source(meta)}`\n\n{snippet}"
            )

    # --- Word ---
    if docx_available():
        st.download_button(
            "⬇️ プロンプト＋回答を Word で保存 (.docx)",
            data=make_bot_answer_docx_bytes(
                prompt_text=prompt_text,
                answer_text=view.answer_text,
                user=current_user or "(anonymous)",
                chat_model=chat_model,
                detail_label=detail_label,
                detail=detail,
                max_tokens=MAX_OUTPUT_TOKENS,
                top_k=int(top_k),
                years_sel=years_sel,
                pnos_sel_norm=pnos_sel_norm,
                shards=[],
            ),
            file_name=make_default_docx_filename(prefix="bot_answer"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    # --- コスト ---
    note = "（Gemini推定）" if view.used_gemini else ""
    st.info(f"📊 API使用料金（概算）: **¥{view.cost_jpy:,.2f}**（${view.cost_usd:.4f}）{note}")

    # --- answer log ---
    logger.append({
        "user": current_user or "(anonymous)",
        "action": "answer",
        "chat_model": chat_model,
        "detail": detail,
        "embedding_tokens": view.embedding_tokens,
        "chat_input_tokens": view.chat_input_tokens,
        "chat_output_tokens": view.chat_output_tokens,
        "cost_usd": view.cost_usd,
        "cost_jpy": view.cost_jpy,
        "prompt_hash": sha256_short(prompt_text),
        "answer_hash": sha256_short(view.answer_text),
        "answer_preview": preview_text(view.answer_text, 20),
    })

    insert_bot_log_row(
        _SQLITE_DB_PATH,
        app=_APP_DIR.name,
        page=_PAGE_NAME,
        user=current_user or "(anonymous)",
        action="answer",
        model=chat_model,
        detail=detail,
        embedding_tokens=view.embedding_tokens,
        input_tokens=view.chat_input_tokens,
        output_tokens=view.chat_output_tokens,
        cost_usd=view.cost_usd,
        cost_jpy=view.cost_jpy,
        prompt_hash=sha256_short(prompt_text),
        answer_hash=sha256_short(view.answer_text),
        prompt=prompt_text if INCLUDE_FULL_PROMPT_IN_LOG else None,
        answer=view.answer_text if INCLUDE_FULL_ANSWER_IN_LOG else None,
    )

else:
    st.info("質問を入力して『送信』を押してください。")
