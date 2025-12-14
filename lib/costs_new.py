# lib/costs_new.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import math
import datetime as dt

import streamlit as st

# ============================================================
# 設定はすべて config から import（single source of truth）
# ============================================================
from config.config import (
    DEFAULT_USDJPY,
    MODEL_PRICES_USD,
    EMBEDDING_PRICES_USD,
    AUDIO_PRICES_USD_PER_MIN,
    MILLION,
)

# ============================================================
# Dataclass
# ============================================================
@dataclass
class ChatUsage:
    input_tokens: int
    output_tokens: int

# ============================================================
# 基本ユーティリティ
# ============================================================
def usd_to_jpy(usd: float, rate: float = DEFAULT_USDJPY) -> float:
    return round(float(usd) * float(rate), 2)

def estimate_chat_cost(
    model: str,
    usage: ChatUsage,
    *,
    rate: float = DEFAULT_USDJPY,
) -> Dict[str, float]:
    price = MODEL_PRICES_USD.get(model)
    if not price:
        raise ValueError(f"単価未設定のモデル: {model}")

    usd = (
        (usage.input_tokens * price["in"])
        + (usage.output_tokens * price["out"])
    ) / MILLION
    usd = round(float(usd), 6)
    return {"usd": usd, "jpy": usd_to_jpy(usd, rate)}

def estimate_embedding_cost(
    model: str,
    input_tokens: int,
    *,
    rate: float = DEFAULT_USDJPY,
) -> Dict[str, float]:
    price = EMBEDDING_PRICES_USD.get(model)
    if price is None:
        raise ValueError(f"単価未設定の埋め込みモデル: {model}")

    usd = round((max(0, int(input_tokens)) / MILLION) * price, 6)
    return {"usd": usd, "jpy": usd_to_jpy(usd, rate)}

def estimate_transcribe_cost(
    model: str,
    seconds: float,
    *,
    rate: float = DEFAULT_USDJPY,
) -> Dict[str, float]:
    price = AUDIO_PRICES_USD_PER_MIN.get(model)
    if price is None:
        raise ValueError(f"単価未設定の音声モデル: {model}")

    minutes = max(0.0, seconds) / 60.0
    usd = round(price * minutes, 6)
    return {"usd": usd, "jpy": usd_to_jpy(usd, rate)}

# ============================================================
# UI: 使用量サマリ表示（Streamlit依存）
# ============================================================
def _model_prices_per_1k() -> Dict[str, Dict[str, float]]:
    return {
        m: {
            "in": float(p.get("in", 0.0)) / 1000.0,
            "out": float(p.get("out", 0.0)) / 1000.0,
        }
        for m, p in MODEL_PRICES_USD.items()
    }

def render_usage_summary(
    *,
    embedding_model: str,
    embedding_tokens: int,
    chat_model: str,
    chat_prompt_tokens: int,
    chat_completion_tokens: int,
    use_backend_openai: bool,
    title: str = "📊 使用量の概算",
):
    emb_cost = {"usd": 0.0, "jpy": 0.0}
    if embedding_model and embedding_tokens > 0:
        emb_cost = estimate_embedding_cost(embedding_model, embedding_tokens)

    chat_cost = {"usd": 0.0, "jpy": 0.0}
    if use_backend_openai and chat_model:
        chat_cost = estimate_chat_cost(
            chat_model,
            ChatUsage(
                input_tokens=chat_prompt_tokens or 0,
                output_tokens=chat_completion_tokens or 0,
            ),
        )

    total_usd = emb_cost["usd"] + chat_cost["usd"]
    total_jpy = usd_to_jpy(total_usd, DEFAULT_USDJPY)

    st.markdown(f"### {title}")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("合計 (JPY)", f"{total_jpy:,.2f} 円")
        st.caption(f"為替 {DEFAULT_USDJPY:.2f} JPY/USD")

    with c2:
        st.write("**内訳 (USD)**")
        st.write(f"- Embedding: `${emb_cost['usd']:.6f}`")
        if use_backend_openai:
            st.write(
                f"- Chat: `${chat_cost['usd']:.6f}` "
                f"(in={chat_prompt_tokens}, out={chat_completion_tokens})"
            )
        st.write(f"- 合計: `${total_usd:.6f}`")

    with c3:
        per_1k = _model_prices_per_1k()
        emb_per_1k = float(EMBEDDING_PRICES_USD.get(embedding_model, 0.0)) / 1000.0
        chat_in = per_1k.get(chat_model, {}).get("in", 0.0)
        chat_out = per_1k.get(chat_model, {}).get("out", 0.0)
        st.write("**単価 (USD / 1K tok)**")
        st.write(f"- Embedding: `${emb_per_1k:.5f}`（{embedding_model}）")
        st.write(f"- Chat 入力: `${chat_in:.5f}`（{chat_model}）")
        st.write(f"- Chat 出力: `${chat_out:.5f}`（{chat_model}）")

    return {
        "embedding_usd": emb_cost["usd"],
        "chat_usd": chat_cost["usd"],
        "total_usd": total_usd,
        "total_jpy": total_jpy,
    }

# ============================================================
# meta.jsonl からの安全な埋め込みコスト概算
# ============================================================
def _percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(math.ceil(q * len(s)) - 1)))
    return float(s[idx])

def _parse_iso_dt(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

def summarize_embedding_cost_from_meta(
    meta_path: Path,
    model: str = "text-embedding-3-large",
    *,
    rate: float = DEFAULT_USDJPY,
    outlier_tok_threshold: int = 8192,
    include_source_paths: Optional[List[str]] = None,
    created_after_iso: Optional[str] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    tokens: List[int] = []
    skipped_outliers = 0
    missing_tokens_rows = 0

    if not meta_path.exists():
        return {
            "model": model,
            "price_per_1M": EMBEDDING_PRICES_USD.get(model, 0.0),
            "rate": rate,
            "total_tokens": 0,
            "n_chunks": 0,
            "avg_tok": 0.0,
            "p95_tok": 0.0,
            "max_tok": 0,
            "skipped_outliers": 0,
            "missing_tokens_rows": 0,
            "warnings": ["meta.jsonl が見つかりません"],
            "usd": 0.0,
            "jpy": 0.0,
        }

    created_after_dt = _parse_iso_dt(created_after_iso)

    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if include_source_paths:
                src = obj.get("source_path") or obj.get("path") or obj.get("file")
                if not src or not any(src.endswith(p) for p in include_source_paths):
                    continue

            if created_after_dt:
                ca = _parse_iso_dt(obj.get("created_at"))
                if ca is None or ca < created_after_dt:
                    continue

            if "chunk_len_tokens" not in obj:
                missing_tokens_rows += 1
                continue

            tok = int(obj.get("chunk_len_tokens", 0))
            if tok <= 0:
                continue
            if outlier_tok_threshold and tok > outlier_tok_threshold:
                skipped_outliers += 1
                continue

            tokens.append(tok)

    if missing_tokens_rows:
        warnings.append(f"`chunk_len_tokens` 欠損行を {missing_tokens_rows} 件スキップしました。")
    if skipped_outliers:
        warnings.append(f"外れ値チャンク（>{outlier_tok_threshold} tok）を {skipped_outliers} 件スキップしました。")

    total_tokens = sum(tokens)
    n = len(tokens)

    est = estimate_embedding_cost(model, total_tokens, rate=rate)

    return {
        "model": model,
        "price_per_1M": EMBEDDING_PRICES_USD.get(model, 0.0),
        "rate": rate,
        "total_tokens": total_tokens,
        "n_chunks": n,
        "avg_tok": (total_tokens / n) if n else 0.0,
        "p95_tok": _percentile(tokens, 0.95) if n else 0.0,
        "max_tok": max(tokens) if n else 0,
        "skipped_outliers": skipped_outliers,
        "missing_tokens_rows": missing_tokens_rows,
        "warnings": warnings,
        "usd": est["usd"],
        "jpy": est["jpy"],
    }
