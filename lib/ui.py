import streamlit as st

def hide_deploy_button():
    st.markdown("""
    <style>
    /* div[data-testid="stDeployButton"] { display: none !important; } */
    /* ツールバーごと隠す場合（メニュー等も消える） */
    div[data-testid="stToolbar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

def warn_before_close():
    """ブラウザを閉じる・リロードする前に確認ダイアログを出す"""
    st.markdown("""
    <script>
    window.addEventListener("beforeunload", function (e) {
        e.preventDefault();
        e.returnValue = '';  // Chrome対応
    });
    </script>
    """, unsafe_allow_html=True)
