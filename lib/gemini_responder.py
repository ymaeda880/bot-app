# lib/gemini_responder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterator

from config.config import get_gemini_api_key

@dataclass
class GeminiUsage:
    # Geminiはusageを取れない/取りにくい場合があるので、基本は推定用（呼び出し側で入れる）
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass
class GeminiResult:
    text: str
    usage: GeminiUsage

class GeminiResponder:
    """
    google-generativeai を使う最小 responder。
    - stream(): 逐次テキストyield
    - complete(): 一括で text 返却
    """
    def __init__(self) -> None:
        self.final_text: str = ""
        self.usage = GeminiUsage()

        key = get_gemini_api_key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY が未設定です（st.secrets）。")

        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError("google-generativeai が未導入です。`pip install google-generativeai` を実行してください。") from e

        genai.configure(api_key=key)
        self._genai = genai

    def complete(
        self,
        *,
        model: str,
        system_instruction: str,
        user_content: str,
        max_output_tokens: int,
    ) -> GeminiResult:
        m = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction,
        )
        resp = m.generate_content(
            user_content,
            generation_config={
                "max_output_tokens": int(max_output_tokens),
            },
            stream=False,
        )

        text = getattr(resp, "text", "") or ""
        self.final_text = text
        return GeminiResult(text=text, usage=self.usage)

    def stream(
        self,
        *,
        model: str,
        system_instruction: str,
        user_content: str,
        max_output_tokens: int,
        on_error_text: str = "Gemini stream error.",
    ) -> Iterator[str]:
        m = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction,
        )
        try:
            chunks = m.generate_content(
                user_content,
                generation_config={
                    "max_output_tokens": int(max_output_tokens),
                },
                stream=True,
            )
            buf = []
            for ch in chunks:
                t = getattr(ch, "text", "") or ""
                if t:
                    buf.append(t)
                    yield t
            self.final_text = "".join(buf)
        except Exception:
            yield on_error_text
            self.final_text = on_error_text
