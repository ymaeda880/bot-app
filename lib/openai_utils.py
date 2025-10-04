# lib/openai_utils.py
from __future__ import annotations
from typing import List
import re
from openai import OpenAI

"""
openai_utils
============

OpenAI API 呼び出しのための軽量ヘルパー群。

主な機能
- tiktoken を用いた概算トークン数の計測（count_tokens）
- 指定トークン数での文字列トリム（truncate_by_tokens）
- GPT-5 系判定ユーティリティ（is_gpt5）
- Chat Completions API を「max_tokens / max_completion_tokens」の両対応で安全に叩く（chat_complete_safely）
- Chat Completions 応答から本文テキストを抽出（extract_text_from_chat）
- Responses API（responses.create）の簡易ラッパー（responses_generate, responses_text）

注意
- tiktoken が未インストール / 未対応モデルの場合は粗い近似にフォールバックします。
- OpenAI の各モデルは、受け付けるパラメータ（例: temperature, max_tokens 系）が異なる場合があります。
  本モジュールでは例外発生時にいくつかフォールバックしますが、将来の API 変更ですべてを吸収できる保証はありません。
"""

# ---- token helpers ----
def _encoding_for(model_hint: str):
    """
    指定モデル名（model_hint）に最適な tiktoken エンコーディングを返す。

    失敗時（未知モデル／未インストール等）は "cl100k_base" を返す。

    Parameters
    ----------
    model_hint : str
        推定したいモデル名（例: "gpt-5-mini", "gpt-4.1" など）

    Returns
    -------
    tiktoken.Encoding
        tiktoken のエンコーディングオブジェクト
    """
    import tiktoken
    try:
        return tiktoken.encoding_for_model(model_hint)
    except Exception:
        # 未知モデルや例外時は cl100k_base にフォールバック
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model_hint: str = "gpt-5-mini") -> int:
    """
    与えられた文字列のトークン数を概算する。

    tiktoken による厳密なカウントを優先し、例外時は「文字数/4」を
    目安として用いる（最低 1 トークン）。

    Parameters
    ----------
    text : str
        対象テキスト
    model_hint : str, default "gpt-5-mini"
        tiktoken のモデル推定に用いるヒント

    Returns
    -------
    int
        推定トークン数
    """
    try:
        enc = _encoding_for(model_hint)
        return len(enc.encode(text or ""))
    except Exception:
        # ざっくり 4 文字 ≒ 1 トークン の近似
        return max(1, int(len(text or "") / 4))


def truncate_by_tokens(text: str, max_tokens: int, model_hint: str = "gpt-5-mini") -> str:
    """
    テキストを「最大トークン数」以内に収まるように切り詰める。

    tiktoken が使える場合はトークンベースで厳密に切り詰め、
    それ以外は「max_tokens*4」文字（最低 100 文字）までの文字列切り詰めに
    フォールバックする。

    Parameters
    ----------
    text : str
        対象テキスト
    max_tokens : int
        許容する最大トークン数
    model_hint : str, default "gpt-5-mini"
        tiktoken のモデル推定に用いるヒント

    Returns
    -------
    str
        切り詰め後のテキスト
    """
    try:
        enc = _encoding_for(model_hint)
        toks = enc.encode(text or "")
        if len(toks) <= max_tokens:
            return text or ""
        # 先頭 max_tokens 個のトークンだけ残して decode
        return enc.decode(toks[:max_tokens])
    except Exception:
        # フォールバック: 文字数ベースで適当に丸める
        max_chars = max(100, max_tokens * 4)
        return (text or "")[:max_chars]


def is_gpt5(model_name: str) -> bool:
    """
    モデル名が GPT-5 系かどうかを判定する（接頭辞 "gpt-5" で判定）。

    Parameters
    ----------
    model_name : str
        判定対象のモデル名

    Returns
    -------
    bool
        True なら GPT-5 系とみなす
    """
    return (model_name or "").lower().startswith("gpt-5")


# ---------- OpenAI 呼び出し ----------
def chat_complete_safely(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    limit_tokens: int,
    system_prompt: str,
    user_prompt: str
):
    """
    Chat Completions API を「max_tokens / max_completion_tokens」両対応で安全に実行する。

    背景
    ----
    - 一部モデル（特に GPT-5 系や o 系）は `max_completion_tokens` を要求／推奨することがあります。
    - それ以外のモデルでは `max_tokens` が一般的です。
    - 本関数では、モデル名ヒューリスティクスでまず片方を試し、失敗したらもう一方で再試行します。

    Parameters
    ----------
    client : OpenAI
        OpenAI クライアント
    model : str
        モデル名
    temperature : float
        生成の多様性（0～2 程度）。※ 一部モデルでは未対応の可能性あり
    limit_tokens : int
        出力トークン上限として渡す値（max_tokens / max_completion_tokens のいずれかに割り当て）
    system_prompt : str
        system ロールのプロンプト（前処理で strip します）
    user_prompt : str
        user ロールのプロンプト

    Returns
    -------
    openai.types.chat.completion.ChatCompletion
        Chat Completions のレスポンスオブジェクト

    Notes
    -----
    - 例外時はパラメータ切り替えで 1 回だけリトライします。
    - モデルの仕様変更により temperature が拒否されるケースもあり得ます。
      その場合は呼び出し側で 0 固定／未指定などの運用を検討してください。
    """
    def _call(use_mct: bool):
        # 共通ペイロードを構築
        payload = {
            "model": model,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
        }
        # モデルに合わせて max_* パラメータ名を切り替える
        if use_mct:
            payload["max_completion_tokens"] = int(limit_tokens)
        else:
            payload["max_tokens"] = int(limit_tokens)
        # 実行
        return client.chat.completions.create(**payload)

    # GPT-5 / o 系（例: o3, o4-mini）っぽい名前なら first-try を max_completion_tokens 側に
    prefer_mct = (model.lower().startswith("gpt-5") or model.lower().startswith("o"))
    try:
        return _call(prefer_mct)
    except Exception:
        # 失敗したらもう片方で再試行
        return _call(not prefer_mct)


def extract_text_from_chat(resp_obj) -> str:
    """
    Chat Completions 応答オブジェクトから本文テキストを安全に取り出す。

    Parameters
    ----------
    resp_obj : Any
        Chat Completions のレスポンス（choices[0].message.content を期待）

    Returns
    -------
    str
        取得できた本文テキスト。失敗時は空文字列。
    """
    try:
        content = resp_obj.choices[0].message.content
        return content or ""
    except Exception:
        return ""


def responses_generate(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
    system_prompt: str,
    user_prompt: str
):
    """
    Responses API（client.responses.create）の簡易ラッパー。

    Chat Completions ではなく Responses API 系モデルを使う場合に利用。
    入力は role 分離した messages 風の配列（system / user）。

    Parameters
    ----------
    client : OpenAI
        OpenAI クライアント
    model : str
        モデル名（例: "gpt-5", "gpt-4.1" など）
    temperature : float
        生成の多様性（※ 一部モデルでは未対応の可能性あり）
    max_output_tokens : int
        出力トークン上限
    system_prompt : str
        system ロールのプロンプト（前処理で strip します）
    user_prompt : str
        user ロールのプロンプト

    Returns
    -------
    openai.types.responses.response.Response
        Responses API のレスポンスオブジェクト
    """
    return client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
    )


def responses_text(resp) -> str:
    """
    Responses API の応答からテキストを取り出す。

    Parameters
    ----------
    resp : Any
        Responses API のレスポンス（resp.output_text を優先的に期待）

    Returns
    -------
    str
        取り出した本文テキスト。取得できない場合は空文字列。
    """
    try:
        txt = resp.output_text
        if isinstance(txt, str) and txt.strip():
            return txt
    except Exception:
        pass
    return ""
