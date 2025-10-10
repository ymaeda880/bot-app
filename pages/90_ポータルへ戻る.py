# pages/90_ポータルへ戻る.py
# ============================================================
# 🏠 ポータルへ戻る
# - Nginx 経由で "/" にアクセス（index.html が自動表示される想定）
# ============================================================

import streamlit as st
import webbrowser

st.set_page_config(page_title="ポータルへ戻る", page_icon="🏠", layout="centered")
st.title("🏠 ポータルへ戻る")

st.markdown(
    """
    ### 🌐 社内ポータルを開く（トップ `/` へ）
    - このボタンを押すと、Nginx のルート `/`（index.html）をブラウザで開きます。  
    - LAN 内からアクセス可能な URL 例:
        - http://localhost/
        - http://<Mac の IP アドレス>/
        - http://<ホスト名.local>/
    """
)

if st.button("🚀 ポータルへ戻る（/ を開く）", type="primary", use_container_width=True):
    try:
        webbrowser.open_new_tab("http://localhost/")
        st.success("ブラウザでポータル（/）を開きました ✅")
    except Exception as e:
        st.error(f"ブラウザを開けませんでした: {e}")

st.caption("補足: LAN 内の他端末は `http://<Mac-IP>/` または `http://<ホスト名.local>/` でアクセス可能です。")
