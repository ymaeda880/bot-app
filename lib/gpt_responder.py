# lib/gpt_responder.py
# ============================================
# OpenAI Responses API の呼び出しを共通化
# - ストリーミング / 一括表示の両対応
# - ストリーミング不可（未検証組織など）の場合は自動フォールバック
# - Streamlit 非依存（任意の Python アプリから利用可能）
# ============================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Generator, Optional, Dict, Any, Iterable

import os
from openai import OpenAI, BadRequestError


@dataclass
class Usage:
    """トークン消費量を表す簡易データ構造。"""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CompletionResult:
    """一括生成の戻り値。"""
    text: str
    usage: Usage


class GPTResponder:
    """
    OpenAI Responses API 呼び出しの薄いラッパ。

    - stream(): 逐次トークン（またはテキスト差分）を yield するジェネレータを返す
      * ストリーム終了後に self.final_text / self.usage が埋まる
      * ストリーミング不可エラー時は自動で非ストリームにフォールバックして全文を一括で yield

    - complete(): 一括で完了させ、CompletionResult を返す

    どちらも Streamlit 非依存で利用できます（GUI層ではこの上に st.write_stream などを被せてください）。
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Args:
            api_key: OpenAI API キー。None の場合は環境変数 OPENAI_API_KEY を利用。
        """
        if api_key and ("OPENAI_API_KEY" not in os.environ):
            os.environ["OPENAI_API_KEY"] = api_key
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY") or "")

        # ストリーム完了後に参照可能
        self.final_text: str = ""
        self.usage: Usage = Usage()

    # ---------- ユーティリティ ----------

    @staticmethod
    def _safe_extract_text(resp: Any) -> str:
        """SDK 差分に強いテキスト抽出。"""
        try:
            return resp.output_text
        except Exception:
            try:
                return resp.output[0].content[0].text
            except Exception:
                return str(resp)

    @staticmethod
    def _usage_from(resp: Any) -> Usage:
        """Usage を安全に取り出す。"""
        u = getattr(resp, "usage", None)
        if not u:
            return Usage()
        return Usage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        )

    # ---------- 一括 ----------

    def complete(
        self,
        *,
        model: str,
        system_instruction: str,
        user_content: str,
        max_output_tokens: Optional[int] = None,
    ) -> CompletionResult:
        """
        一括でレスポンスを取得する。

        Args:
            model: 使用モデル（例: "gpt-5-mini", "gpt-4.1"）
            system_instruction: system ロールのテキスト
            user_content: user ロールのテキスト（完成済みのプロンプト）
            max_output_tokens: 出力トークン上限

        Returns:
            CompletionResult: text と usage（input/output tokens）
        """
        kwargs = dict(
            model=model,
            input=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
        )
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)

        resp = self._client.responses.create(**kwargs)
        text = self._safe_extract_text(resp)
        usage = self._usage_from(resp)
        self.final_text = text
        self.usage = usage
        return CompletionResult(text=text, usage=usage)

    # ---------- ストリーム（逐次） ----------

    def stream(
        self,
        *,
        model: str,
        system_instruction: str,
        user_content: str,
        max_output_tokens: Optional[int] = None,
        on_error_text: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        逐次出力（トークン／デルタ）を yield する。

        - ストリーミング不可エラー（BadRequestError/400 + stream unsupported）時は
          非ストリームで生成し、全文を一括で yield する（フォールバック）。
        - ストリーム完了後、self.final_text と self.usage が設定される。

        Args:
            model: 使用モデル
            system_instruction: system ロール
            user_content: user ロール
            max_output_tokens: 出力トークン上限
            on_error_text: ストリーム中の error イベントに併記するメッセージ（任意）

        Yields:
            str: 逐次出力されるテキスト断片（デルタ）
        """
        kwargs = dict(
            model=model,
            input=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
        )
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)

        try:
            # ストリーミング開始
            with self._client.responses.stream(**kwargs) as stream:
                for event in stream:
                    t = getattr(event, "type", "")
                    if t.endswith(".delta"):
                        delta = getattr(event, "delta", None)
                        if isinstance(delta, str) and delta:
                            yield delta
                        else:
                            tok = getattr(event, "token", None)
                            if isinstance(tok, str) and tok:
                                yield tok
                    elif t == "response.error":
                        err = getattr(event, "error", None)
                        if err:
                            yield f"\n\n[ERROR] {on_error_text or ''} {err}".strip()

                # ストリーム終了：最終レスポンスから全文と usage を取得
                final = stream.get_final_response()
                self.final_text = getattr(final, "output_text", "")
                self.usage = self._usage_from(final)

        except BadRequestError as e:
            # よくあるケース：未検証Orgで stream 禁止（unsupported_value/param=stream 等）
            msg = str(getattr(e, "response", None) or e)
            if (
                "must be verified to stream" in msg
                or "param': 'stream'" in msg
                or "unsupported_value" in msg
            ):
                # 非ストリームでフォールバック
                result = self.complete(
                    model=model,
                    system_instruction=system_instruction,
                    user_content=user_content,
                    max_output_tokens=max_output_tokens,
                )
                # 呼び出し側が write_stream 等にそのまま渡せるよう、全文を一括で yield
                yield result.text
            else:
                # 想定外はそのまま投げる
                raise
