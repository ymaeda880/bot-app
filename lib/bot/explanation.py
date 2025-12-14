# lib/bot/explanation.py
from __future__ import annotations
import streamlit as st


def render_bot_usage_expander(*, expanded: bool = False) -> None:
    with st.expander("ℹ️ このページの使い方", expanded=expanded):
        st.markdown("""
### 1) 何ができる？
- 社内PDFをベクトルDBから **RAG 検索＋回答生成** します。
- **year / pno フィルタ** で範囲を絞れます。
- **OpenAI / Gemini をモデル選択で切り替え可能**です（Gemini は API Key 設定時のみ表示）。
- 生成した **質問＋回答** を **Word（.docx）** で保存できます（フィルタ情報つき）。
- 出典は **[S1]** 形式で提示し、必要に応じて出典一覧を展開します。

### 2) サイドバー（主な設定）
- **モデル**、**検索件数（Top-K）**、**詳しさ**、**最大出力トークン**、**System Instruction**、**表示モード（逐次/一括）**、**year / pno フィルタ**。

### 3) 使い方の流れ
1. サイドバーで必要に応じて設定（year/pno など）
2. 入力欄に質問を入れて **送信**
3. 回答の下に **出典（[S1]…）** と **参照コンテキスト** を表示
4. 必要に応じて **Word で保存** ボタンから .docx を出力
        """)
