# pages/50_ポータルへ戻る.py
# ============================================================
# 🔙 /auth_portal/ へ戻るページ
# ============================================================

import streamlit as st

st.set_page_config(page_title="戻る（auth_portal）", page_icon="🔙", layout="centered")

st.title("🔙 認証ポータルへ戻る")

st.write("このページから、ログインポータル（/auth_portal/）に戻ることができます。")

# Nginx経由で動いている前提（/auth_portal/ がルートURLの一部として有効）
portal_url = "/auth_portal/"

st.markdown(
    f"""
    <div style='margin-top:2em; text-align:center;'>
        <a href="{portal_url}" target="_self" style="
            display:inline-block;
            padding: 0.6em 1.2em;
            font-size:1.1em;
            color:white;
            background-color:#1E88E5;
            border-radius:8px;
            text-decoration:none;
        ">
            🔐 認証ポータルへ戻る
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
