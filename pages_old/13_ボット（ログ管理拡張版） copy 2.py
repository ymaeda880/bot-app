# pages/13_ボット（ログ管理拡張版）.py
from __future__ import annotations

from pathlib import Path
from typing import Set
import sys
import re
import inspect

import streamlit as st

# --- sys.path 調整 ---
_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

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

from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie
from common_lib.logs.jsonl_logger import JsonlLogger, sha256_short

from lib.logs.sqlite_logger import (
    init_bot_logs_db,
    insert_bot_log_row,
    preview_text,
)

from common_lib.auth.config import COOKIE_NAME
from lib.debug.auth_debug import render_auth_debug


with st.sidebar:
    st.header("設定")
    # …既存の sidebar UI …

    # ===== 一時的なデバッグ表示 =====
    render_auth_debug(
        get_user_func=get_current_user_from_session_or_cookie,
        cookie_name=COOKIE_NAME,
    )

# ============================================================
# Logger（JSONL：管理者監査用）
# ============================================================

_APP_DIR = Path(__file__).resolve().parents[1]
_PAGE_NAME = Path(__file__).stem

logger = JsonlLogger(
    app_dir=_APP_DIR,
    page_name=_PAGE_NAME,
    rotate="monthly",
)

INCLUDE_FULL_PROMPT_IN_LOG = True
INCLUDE_FULL_ANSWER_IN_LOG = True

# ============================================================
# Models / Constants
# ============================================================

OPENAI_MODELS = ["gpt-5-mini", "gpt-5-nano"]
GEMINI_MODELS = ["gemini-2.0-flash"]

MAX_OUTPUT_TOKENS = 12_000

# ============================================================
# helpers
# ============================================================

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
) -> BotAnswerView:
    """vectorspace 対応の有無を吸収する互換呼び出し"""
    try:
        sig = inspect.signature(run_bot_query)
        if "vectorspace" in sig.parameters:
            return run_bot_query(
                question=question,
                chat_model=chat_model,
                detail=detail,
                max_tokens=max_tokens,
                top_k=top_k,
                years_sel=years_sel,
                pnos_sel_norm=pnos_sel_norm,
                system_instruction=system_instruction,
                vectorspace=vectorspace,
            )
    except Exception:
        pass

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

def get_user_sqlite_path(project_root: Path, user_sub: str) -> Path:
    """
    利用者履歴DB（SQLite）の正本パス
    projects/Storages/<sub>/bot_app/logs/bot_logs.db
    """
    return (
        project_root
        / "Storages"
        / user_sub
        / "bot_app"
        / "logs"
        / "bot_logs.db"
    )

# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="Chat Bot — Pipeline版",
    page_icon="💬",
    layout="wide",
)

user, _ = get_current_user_from_session_or_cookie(st)

# user は str（例: "maeda"）で返るのが正
user_sub = user if isinstance(user, str) else None

col_title, col_user = st.columns([5, 2], vertical_alignment="center")
with col_title:
    st.title("💬 社内ボット")
with col_user:
    if user_sub:
        st.success(f"ログイン中: **{user_sub}**")
    else:
        st.warning("未ログイン")

st.caption("このアプリは、操作ログ・回答ログを記録します。")

# ログイン必須
if not user_sub:
    st.error("ログインが必要です。")
    st.stop()


# ============================================================
# ログイン必須（SQLite用途）
# ============================================================

if not user_sub:
    st.sidebar.write("session type =", type(session).__name__)
    st.sidebar.write("session repr =", repr(session)[:300])
    st.sidebar.write("user_sub =", user_sub)
    st.error("ログインが必要です。")
    st.stop()


# ============================================================
# project_root 解決（settings.toml）
# ============================================================

from config.path_config import PATHS
project_root = Path(PATHS.project_root)

# ============================================================
# SQLite 初期化（利用者専用）
# ============================================================

_sqlite_path = get_user_sqlite_path(project_root, user_sub)
_sqlite_path.parent.mkdir(parents=True, exist_ok=True)
_SQLITE_DB_PATH = init_bot_logs_db(_sqlite_path)

# ============================================================
# 状態
# ============================================================

if "q" not in st.session_state:
    st.session_state.q = ""

def _set_q(x: str) -> None:
    st.session_state.q = x or ""

render_bot_usage_expander(expanded=False)
st.divider()

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("設定")

    st.subheader("📚 検索DB（vectorstore）")
    vectorspace_label = st.radio(
        "対象",
        ["report（openai）", "規定集（openai_sample）"],
        index=0,
    )
    VECTORSPACE_MAP = {
        "report（openai）": "openai",
        "規定集（openai_sample）": "openai_sample",
    }
    vectorspace = VECTORSPACE_MAP[vectorspace_label]

    st.divider()

    model_options = list(OPENAI_MODELS)
    if has_gemini_api_key():
        model_options += GEMINI_MODELS

    chat_model = st.radio("モデル", model_options, index=0)
    top_k = st.slider("検索件数（Top-K）", 1, 12, 6, 1)

    detail_label = st.selectbox("詳しさ", ["簡潔", "標準", "詳細", "超詳細"], index=2)
    detail = {
        "簡潔": "concise",
        "標準": "standard",
        "詳細": "detailed",
        "超詳細": "very_detailed",
    }[detail_label]

    system_instruction = "あなたは優秀な社内のアシスタントです."

    st.divider()
    st.subheader("year / pno フィルタ")

    years_sel = parse_years(st.text_input("year（任意）", ""))
    pnos_raw = parse_pnos(st.text_input("pno（任意）", ""))
    pnos_sel_norm: Set[str] = set()
    for p in pnos_raw:
        pnos_sel_norm |= norm_pno_forms(p)

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
    page_for_sqlite = f"{_PAGE_NAME}__{vectorspace}"

    # --- ask log (jsonl：管理者用) ---
    logger.append({
        "user": user_sub,
        "action": "ask",
        "vectorspace": vectorspace,
        "chat_model": chat_model,
        "detail": detail,
        "top_k": int(top_k),
        "max_tokens": MAX_OUTPUT_TOKENS,
        "prompt_hash": sha256_short(prompt_text),
        **({"prompt": prompt_text} if INCLUDE_FULL_PROMPT_IN_LOG else {}),
    })

    # --- ask log (sqlite：利用者履歴) ---
    insert_bot_log_row(
        _SQLITE_DB_PATH,
        app=_APP_DIR.name,
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

    view: BotAnswerView = _run_bot_query_compat(
        question=prompt_text,
        chat_model=chat_model,
        detail=detail,
        max_tokens=MAX_OUTPUT_TOKENS,
        top_k=int(top_k),
        years_sel=years_sel,
        pnos_sel_norm=pnos_sel_norm,
        system_instruction=system_instruction,
        vectorspace=vectorspace,
    )

    st.subheader("🧠 回答")
    st.write(view.answer_text)

    # --- answer log (jsonl) ---
    logger.append({
        "user": user_sub,
        "action": "answer",
        "vectorspace": vectorspace,
        "chat_model": chat_model,
        "detail": detail,
        "prompt_hash": sha256_short(prompt_text),
        "answer_hash": sha256_short(view.answer_text),
        "answer_preview": preview_text(view.answer_text, 20),
    })

    # --- answer log (sqlite) ---
    insert_bot_log_row(
        _SQLITE_DB_PATH,
        app=_APP_DIR.name,
        page=page_for_sqlite,
        user=user_sub,
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
