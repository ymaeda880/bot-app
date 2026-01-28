# lib/prompts/bot_prompt.py
# ============================================
# 社内ボット（RAG）の厳格プロンプトを生成するユーティリティ
# - “資料外は分かりません” を強制（strict mode）
# - スタイル（style hint）のプリセット付き
# - 引用表記（[S1], [S2] ...）の明示を指示可能
# ============================================

# ============================================================
# bot prompt builder（RAG 用プロンプト生成）
#
# 【重要な設計メモ】
# 本ファイルは「過去の bot で最も回答品質が高かった
# プロンプト構造」を正本としている。
#
# 過去プロンプトの特徴：
# - prompt 本文の先頭に "# System {sys_inst}" を含める
# - system role と user prompt の責務を厳密に分離しない
# - strict=False を前提とし、
#   「必要に応じて一般知識で補足してよい」挙動を許容
#
# 新実装では以下を試みたが、回答品質が低下した：
# - system / user プロンプトの厳密分離
# - strict=True（資料外禁止）の強化
# - 共通ルール（COMMON_PROMPT_TAIL 等）の追加
#
# そのため現在は、
# 「過去と同一の prompt 文字列を生成する」ことを最優先とし、
# 以下を採用している：
#
# - build_system_instruction() は使用しない
# - prompt 本文に System / Task / Context / Requirements を
#   すべて含める
# - strict=False をデフォルト運用とする
#
# 注意：
# この実装は「厳密な strict RAG」よりも
# 「実務的に読みやすく、説明力のある回答」を優先している。
# 将来 strict RAG に切り替える場合は、
# pipeline.py 側と同時に再設計すること。
# ============================================================



from __future__ import annotations
from typing import List

DEFAULT_SYS_INST = "あなたは社内アシスタントです。"

# スタイルのプリセット（style hints）
STYLE_MAP = {
    "concise": "箇条書きで要点のみ、150-250字程度。",
    "standard": "見出し＋箇条書きで300-500字程度。",
    "detailed": "見出し＋箇条書き＋要約で500-800字程度。",
    "very_detailed": "丁寧な要約と段落構成で800字以上。",
}


def _guard_text(strict: bool) -> str:
    """
    strict=True のとき：
      コンテキスト（retrieved contexts）以外の知識は禁止。
      “この資料からは分かりません” だけを返す／
      提案や一般論の補足もしない（no suggestion, no speculation）。
    """
    if strict:
        return (
            "以下のコンテキストに書かれていること【のみ】を根拠に回答してください。"
            "質問に対応する情報がコンテキストに含まれていない場合、"
            "『この資料からは分かりません』とだけ答えてください。"
            "絶対に一般知識・推測・提案・補足を加えないでください。"
        )
    else:
        return "必要に応じて一般知識で補足しても構いません。"


def build_prompt(
    question: str,
    labeled_contexts: List[str],
    *,
    sys_inst: str = DEFAULT_SYS_INST,
    style_hint: str = "standard",
    cite: bool = True,
    strict: bool = True,
) -> str:
    """
    RAG 回答用のプロンプト（prompt）を生成します。

    Parameters
    ----------
    question : str
        ユーザー質問（user question）
    labeled_contexts : List[str]
        取得済み文脈（retrieved contexts）。
        例: ["[S1] 本文...\n[meta: 2025/a.pdf p.3 / score=0.812]", "[S2] ...", ...]
    sys_inst : str, default DEFAULT_SYS_INST
        system role の指示（system instruction）
    style_hint : {"concise","standard","detailed","very_detailed"}, default "standard"
        出力の粒度・分量（style）
    cite : bool, default True
        回答内に [S1], [S2] などの参照明記を要求（citation line を挿入）
    strict : bool, default True
        厳格モード。コンテキスト外の情報は禁止（no suggestion / no speculation）

    Returns
    -------
    str
        モデルへ渡す最終プロンプト（multi-section text）
    """
    style = STYLE_MAP.get(style_hint, STYLE_MAP["standard"])
    ctx = "\n\n".join(labeled_contexts) if labeled_contexts else "(なし)"
    citeline = "根拠箇所は [S1], [S2] の形式で必ず明記してください。" if cite else ""
    guard = _guard_text(strict)

    prompt = f"""# System
{sys_inst}

# Task
次のユーザー質問に、日本語で回答してください。

# User Question
{question}

# Retrieved Contexts
{ctx}

# Requirements
- {style}
- {citeline}
- {guard}
"""
    return prompt
