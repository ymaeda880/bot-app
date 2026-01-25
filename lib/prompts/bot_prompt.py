# lib/prompts/bot_prompt.py
# ============================================
# 社内ボット（RAG）の厳格プロンプトを生成するユーティリティ
# - “資料外は分かりません” を強制（strict mode）
# - スタイル（style hint）のプリセット付き
# - 引用表記（[S1], [S2] ...）の明示を指示可能
# ============================================

from __future__ import annotations
from typing import List

DEFAULT_SYS_INST = "あなたは社内アシスタントです。"


# ============================================================
# 共通の system 指示（全ボット共通）
# - 振る舞い・姿勢・出力ポリシーのみを書く
# - 知識・事実・補足は禁止（strict RAG を壊さない）
# ============================================================
COMMON_SYSTEM_RULES = """
- 回答は常に日本語で行う。
- 断定できない場合は曖昧にせず、不明と明示する。
- 指示されていない追加説明・提案・一般化は行わない。
""".strip()


# ============================================================
# 共通の prompt 末尾指示（Retrieved Contexts の後に置く）
# - 取得済みコンテキストを読んだ直後に再掲したい実務ルール
# - 内容（知識）の追加は禁止。出力形式・根拠の付け方などに限定
# ============================================================
COMMON_PROMPT_TAIL = """
- 回答は必ず「Retrieved Contexts」の記述を根拠にして書くこと。
- 根拠を示せない文は書かないこと（strict の場合は特に厳守）。
- 資料が空でないときは，回答を行うように努めること．
- 「資料から」，「資料によると」，「〜と記載されている」などという「資料をまとめていることがわかるような」文言は使用しないこと．
- 「〜と記述があります」，「〜と示されています」などという「資料をまとめていることがわかるような」文言は使用しないこと．
- 適切に改行しながら，読みやすい出力にすること．
- ユーザーの質問に対して，適切な日本語で返答すること．

""".strip()







# スタイルのプリセット（style hints）
# STYLE_MAP = {
#     "concise": "箇条書きで要点のみ、150-250字程度。",
#     "standard": "見出し＋箇条書きで300-500字程度。",
#     "detailed": "見出し＋箇条書き＋要約で500-800字程度。",
#     "very_detailed": "丁寧な要約と段落構成で800字以上。",
# }

STYLE_MAP = {
    "concise": "要点を簡潔にまとめる。箇条書き可。目安として100-300字程度。",
    "standard": "文章中心で説明する。見出しや箇条書きに過度に依存しない。目安として300-500字程度。",
    "detailed": (
        "文章中心で説明する。箇条書きは必要な場合にのみ最小限使用する。目安として500-1000字程度．\n"
        "改行を適切に入れながら，読みやすくすること。"
    ),
    "very_detailed": (
        "文章中心で詳述する。箇条書きは原則使わない。目安として1000字以上。．\n"
        "改行を適切に入れながら，読みやすくすること。"
    ),
}


# ============================================================
# 社内標準グロッサリー（Internal Glossary）
# - 記号の意味（解釈ルール）を固定するための定義
# - strict RAG を壊さない（知識追加ではなく用語解釈）
# ============================================================
GLOSSARY_TEXT = ""


#
# システム命令の作成
#
def build_system_instruction(
    *,
    sys_inst: str = DEFAULT_SYS_INST,
    strict: bool = True,
) -> str:
    guard = _guard_text(strict)
    return f"""{sys_inst}

{COMMON_SYSTEM_RULES}

{GLOSSARY_TEXT}

# System Requirements
- {guard}
"""



def _guard_text(strict: bool) -> str:
    """
    strict=True のとき：
      コンテキスト（retrieved contexts）以外の知識は禁止。
      “この資料からは分かりません” だけを返す／
      提案や一般論の補足もしない（no suggestion, no speculation）。

    strict=False のとき：
      まずコンテキストを根拠に回答し、不足分のみ一般知識で補足してよい。
      ただし「資料由来」と「補足（一般知識）」を混ぜない。
    """
    if strict:
        return (
            "以下のコンテキストに書かれていること【のみ】を根拠に回答してください。"
            "質問に対応する情報がコンテキストに含まれていない場合、"
            "『この資料からは分かりません』とだけ答えてください。"
            "絶対に一般知識・推測・提案・補足を加えないでください。"
            "説明的な構成や一般化は行わず、資料に記載された事実・内容・記述を忠実に整理して示す。"
            "背景整理、課題設定、評価、提案などは行わない。"
        )
    else:
        return (
            "まず以下のコンテキストを根拠に回答してください。"
            "コンテキストに情報が不足する場合のみ、一般知識で補足して構いません。"
            "ただし補足は必ず『補足（一般知識）』と明示し、"
            "コンテキスト由来の記述（[S1] 等の根拠）と混ぜないでください。"
            "推測が入る場合は『推測』と明示し、断定しないでください。"
        )


def build_prompt(
    question: str,
    labeled_contexts: List[str],
    *,
    sys_inst: str = DEFAULT_SYS_INST,
    style_hint: str = "standard",
    cite: bool = True,
    strict: bool = True,
) -> str:
    style = STYLE_MAP.get(style_hint, STYLE_MAP["standard"])
    ctx = "\n\n".join(labeled_contexts) if labeled_contexts else "(なし)"
    citeline = "根拠箇所は [S1], [S2] の形式で必ず明記してください。" if cite else ""
    guard = _guard_text(strict)

    prompt = f"""# Task
次のユーザー質問に、日本語で回答してください。

# User Question
{question}

# Retrieved Contexts
{ctx}

# Common Rules (after contexts)
{COMMON_PROMPT_TAIL}

# Requirements
- {style}
- {citeline}
"""
    return prompt
