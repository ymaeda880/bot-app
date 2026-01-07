# lib/debug/auth_debug.py
from __future__ import annotations

import streamlit as st

def render_auth_debug(
    *,
    get_user_func,
    cookie_name: str,
):
    """
    認証まわりの診断を sidebar に表示する。
    - 本体ロジックを汚さない
    - 例外は握りつぶさず可視化
    """
    st.sidebar.markdown("### 🔍 Auth Debug")

    # --- Cookie 可視化 ---
    cookies = st.context.cookies
    st.sidebar.write("COOKIE_NAME =", cookie_name)
    st.sidebar.write("cookies keys =", list(cookies.keys()))
    st.sidebar.write("has cookie? =", cookie_name in cookies)

    token = cookies.get(cookie_name)
    if token:
        st.sidebar.write("cookie head =", token[:30] + "...")
    else:
        st.sidebar.warning("cookie value = None")

    # --- PyJWT ---
    try:
        import jwt
        st.sidebar.success(f"PyJWT OK ({getattr(jwt, '__version__', 'unknown')})")
    except Exception as e:
        st.sidebar.error(f"PyJWT import failed: {e}")
        return  # ここで打ち切ってよい

    # --- get_current_user_from_session_or_cookie ---
    try:
        user, session = get_user_func(st)
        st.sidebar.write("user type =", type(user).__name__)
        st.sidebar.write("user repr =", repr(user))
        st.sidebar.write("session repr =", repr(session))
    except Exception as e:
        st.sidebar.error(f"get_user_func FAILED: {type(e).__name__}: {e}")
