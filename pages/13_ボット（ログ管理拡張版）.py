# ============================================================
# pages/13_ボット（ログ管理拡張版）.py
# 方針：
# - 利用者履歴（SQLite）   : Storages/<sub>/bot_app/ 配下（現状どおり）
# - 管理者監査ログ（JSONL）: Storages/logs/bot_app/ 配下（新設）
#   ※ storage_policy.mode に従い internal/external を自動切替
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Set
import sys
import re
import inspect

import streamlit as st

# ============================================================
# ページ設定（最初に1回だけ）
# ============================================================
st.set_page_config(
    page_title="Chat Bot — Pipeline版",
    page_icon="💬",
    layout="wide",
)

# ============================================================
# sys.path 調整（設計上の確定事項）
# ============================================================
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

from common_lib.auth.auth_helpers import require_login
from common_lib.logs.jsonl_logger import JsonlLogger, sha256_short

from lib.logs.sqlite_logger import (
    init_bot_logs_db,
    insert_bot_log_row,
    preview_text,
)

from common_lib.storage.external_ssd_root import resolve_storage_subdir_root

# ============================================================
# Models / Constants
# ============================================================
OPENAI_MODELS = ["gpt-5-mini", "gpt-5-nano"]
GEMINI_MODELS = ["gemini-2.0-flash"]
MAX_OUTPUT_TOKENS = 12_000

INCLUDE_FULL_PROMPT_IN_LOG = True
INCLUDE_FULL_ANSWER_IN_LOG = True

APP_DIRNAME = "bot_app"
_PAGE_NAME = Path(__file__).stem
_ADMIN_LOG_PREFIX = "bot_app"


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
    """
    pipeline 側が vectorspace 引数に未対応でも落ちないように互換で呼ぶ。
    - 対応していれば vectorspace を渡す
    - 未対応なら従来どおりに呼ぶ（既定のDBを検索）
    """
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


# ============================================================
# タイトル部
# ============================================================
sub = require_login(st)
if not sub:
    st.stop()

# このページ内で使うログインIDは user_sub に統一
user_sub = sub
owner_sub = sub

left, right = st.columns([2, 1])
with left:
    st.title("📝 社内ボット")
with right:
    st.success(f"✅ ログイン中: **{sub}**")

st.caption("このアプリは、操作ログ（管理者監査用JSONL）・回答履歴（利用者用SQLite）を記録します。")

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
base_dir = STORAGE_ROOT / owner_sub / APP_DIRNAME

# mkdir はページ側で明示（ライブラリは mkdir しない）
(base_dir / "logs").mkdir(parents=True, exist_ok=True)

# SQLite 初期化（利用者専用）
_SQLITE_DB_PATH = init_bot_logs_db(base_dir)

# ============================================================
# (B) 管理者監査ログ（JSONL）：Storages/logs/bot_app/
# ============================================================
admin_logs_root = STORAGE_ROOT / "logs" / APP_DIRNAME
admin_logs_root.mkdir(parents=True, exist_ok=True)

logger = JsonlLogger(
    projects_root=PROJECTS_ROOT,   # .../projects
    app_name=APP_DIRNAME,          # "bot_app"
    page_name=_PAGE_NAME,   # 監査ログ側の page_name
    log_name=_ADMIN_LOG_PREFIX,    # ファイル名ベース（任意。省略すると app_name）
    rotate="monthly",
)

# === DEBUG（必要ならコメントアウト）===
st.caption(f"[DEBUG] storages_root     = {STORAGE_ROOT}")
st.caption(f"[DEBUG] user_base_dir     = {base_dir}")
st.caption(f"[DEBUG] admin_logs_root   = {admin_logs_root}")

# ============================================================
# 状態
# ============================================================
if "q" not in st.session_state:
    st.session_state.q = ""


def _set_q(x: str) -> None:
    st.session_state.q = x or ""


# --- 使い方 ---
render_bot_usage_expander(expanded=False)
st.divider()

# ============================================================
# Sidebar（本体UI：ここから）
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

    # SQLite には vectorspace を埋め込んで残す（スキーマ変更なし）
    page_for_sqlite = f"{_PAGE_NAME}__{vectorspace}"

    # --- ask log (jsonl：管理者用) ---
    logger.append({
        "user": user_sub,
        "action": "ask",
        "vectorspace": vectorspace,
        "chat_model": chat_model,
        "detail_label": detail_label,
        "detail": detail,
        "top_k": int(top_k),
        "max_tokens": MAX_OUTPUT_TOKENS,
        "prompt_hash": sha256_short(prompt_text),
        **({"prompt": prompt_text} if INCLUDE_FULL_PROMPT_IN_LOG else {}),
    })

    # --- ask log (sqlite：利用者履歴) ---
    insert_bot_log_row(
        _SQLITE_DB_PATH,
        app=APP_DIRNAME,  # ★固定名に統一（_APP_DIR.name 依存を外す）
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

    # --- pipeline ---
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

    # --- 回答 ---
    st.subheader("🧠 回答")
    st.write(view.answer_text)

    # --- 出典拡張 ---
    citations = re.findall(r"\[S[^\]]+\]", view.answer_text)
    if citations:
        uniq: list[str] = []
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

    # --- コスト ---
    note = "（Gemini推定）" if getattr(view, "used_gemini", False) else ""
    st.info(f"📊 API使用料金（概算）: **¥{view.cost_jpy:,.2f}**（${view.cost_usd:.4f}）{note}")

    # --- answer log (jsonl：管理者用) ---
    logger.append({
        "user": user_sub,
        "action": "answer",
        "vectorspace": vectorspace,
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
        **({"answer": view.answer_text} if INCLUDE_FULL_ANSWER_IN_LOG else {}),
    })

    # --- answer log (sqlite：利用者履歴) ---
    insert_bot_log_row(
        _SQLITE_DB_PATH,
        app=APP_DIRNAME,
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
