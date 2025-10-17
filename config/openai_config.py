# config/openai_config.py

import streamlit as st
from lib.costs import (
    MODEL_PRICES_USD,      # 料金表（100万tokあたり）
    EMBEDDING_PRICES_USD,  # 埋め込み料金（100万tokあたり）
    DEFAULT_USDJPY,        # 為替（secrets で上書き可）
)

# --- OpenAI API エンドポイント & キー取得 ---
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_CHAT_URL       = "https://api.openai.com/v1/chat/completions"

def get_openai_api_key() -> str:
    return st.secrets.get("OPENAI_API_KEY", "")

# --- モデル別の推奨出力上限（目安） ---
_v1 = 128000
MAX_COMPLETION_BY_MODEL = {
    "gpt-5": _v1,
    "gpt-5-mini": _v1,
    "gpt-5-nano": _v1,
    "gpt-4.1": _v1,
    "gpt-4.1-mini": _v1,
    "gpt-4o": _v1,
    "gpt-4o-mini": _v1,
    "gpt-3.5-turbo": 10000,
}

# 参考: 料金の見積りは lib/costs.py の関数を使うこと
# 例:
#   from lib.costs import estimate_embedding_cost, estimate_chat_cost, usd_to_jpy
#   estimate_embedding_cost("text-embedding-3-large", tokens)
