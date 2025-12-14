# config/config.py
from __future__ import annotations

import streamlit as st

# ============================================================
# Secrets / Keys
# ============================================================
def get_openai_api_key() -> str:
    return st.secrets.get("OPENAI_API_KEY", "")

def get_gemini_api_key() -> str:
    return st.secrets.get("GEMINI_API_KEY", "")

def has_gemini_api_key() -> bool:
    return bool(get_gemini_api_key())

# OpenAI Transcribe endpoint（必要なら他ファイルから参照）
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"

# ============================================================
# FX (USDJPY)
# ============================================================
DEFAULT_USDJPY = float(st.secrets.get("USDJPY", 150.0))

# ============================================================
# Constants
# ============================================================
MILLION = 1_000_000

# ============================================================
# Prices (USD / 1M tokens) - Text models (generation/understanding)
#   - OpenAI: 実運用の単価
#   - Gemini: 公式改定があり得るため「概算用」
# ============================================================
MODEL_PRICES_USD = {
    # --- OpenAI ---
    "gpt-5":         {"in": 1.25,  "out": 10.00},
    "gpt-5-mini":    {"in": 0.25,  "out": 2.00},
    "gpt-5-nano":    {"in": 0.05,  "out": 0.40},
    "gpt-4o":        {"in": 2.50,  "out": 10.00},
    "gpt-4o-mini":   {"in": 0.15,  "out": 0.60},
    "gpt-4.1":       {"in": 2.00,  "out": 8.00},   # 参考
    "gpt-4.1-mini":  {"in": 0.40,  "out": 1.60},   # 参考
    "gpt-3.5-turbo": {"in": 0.50,  "out": 1.50},   # 参考

    # --- Gemini（概算用） ---
    "gemini-2.0-flash": {"in": 0.30, "out": 2.50},
    "gemini-2.0-pro":   {"in": 1.25, "out": 10.00},
}

# ============================================================
# Prices (USD / 1M tokens) - Embeddings
# ============================================================
EMBEDDING_PRICES_USD = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,  # レガシー
}

# ============================================================
# Prices (USD / minute) - Audio
# ============================================================
WHISPER_PRICE_PER_MIN = 0.006

AUDIO_PRICES_USD_PER_MIN = {
    "whisper-1": WHISPER_PRICE_PER_MIN,
}

# もし “分単価” のモデル（OpenAI Transcribe系など）も扱うならここに集約
TRANSCRIBE_PRICES_USD_PER_MIN = {
    "gpt-4o-mini-transcribe": 0.0125,
    "gpt-4o-transcribe":      0.025,
    "whisper-1":              WHISPER_PRICE_PER_MIN,
    # Gemini は「分単価」ではない想定
}

# ============================================================
# Gemini (and generic) token estimation utility
#   厳密ではないが費用目安には十分。
#   目安：1 token ≒ 4 characters（日本語含む概算）
# ============================================================
def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(len(text) // 4, 1)

# ============================================================
# Helpers (optional)
# ============================================================
def is_gemini_model(model: str) -> bool:
    return (model or "").startswith("gemini-")

def is_openai_model(model: str) -> bool:
    return not is_gemini_model(model)
