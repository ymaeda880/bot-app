# pages/100_ログインテスト.py  （bot_app 用）
from __future__ import annotations

from pathlib import Path
import sys
import json
import os

import streamlit as st

# ============================================================
# sys.path 調整（pages/13_ボット に倣う）
# ============================================================
_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

# ============================================================
# 依存：auth helpers / config / jwt_utils
#   ※ common_lib は変更しない（読むだけ）
# ============================================================
from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie

# config / jwt_utils を「どのファイルから読んでいるか」を必ず出す
try:
    import common_lib.auth.config as auth_cfg
    _auth_cfg_ok = True
    _auth_cfg_err = None
except Exception as e:
    auth_cfg = None  # type: ignore
    _auth_cfg_ok = False
    _auth_cfg_err = repr(e)

try:
    import common_lib.auth.jwt_utils as jwt_utils
    _jwt_ok = True
    _jwt_err = None
except Exception as e:
    jwt_utils = None  # type: ignore
    _jwt_ok = False
    _jwt_err = repr(e)

# extra_streamlit_components が入っているか（Cookie確認用）
try:
    import extra_streamlit_components as stx  # type: ignore
    _stx_ok = True
except Exception as e:
    stx = None  # type: ignore
    _stx_ok = False
    _stx_err = repr(e)


def _safe_preview(x, n: int = 10) -> str | None:
    if x is None:
        return None
    s = str(x)
    return (s[:n] + "...") if len(s) > n else s


def _debug_cookie_present(cookie_name: str) -> tuple[bool, str | None]:
    """
    CookieManager 経由で cookie_name が読めるか確認（値は先頭だけ）。
    """
    if not _stx_ok or stx is None:
        return False, None
    try:
        cm = stx.CookieManager(key="cm_login_test_bot")
        v = cm.get(cookie_name)
        if isinstance(v, str) and v:
            return True, (v[:12] + "...")
        return False, None
    except Exception:
        return False, None


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="ログインテスト（bot）", page_icon="🧪", layout="centered")
st.title("🧪 ログインテスト（bot_app）")
st.caption("minutes_app と同じ観点で、Cookie/JWT/config/環境変数の差を特定します。")

# ============================================================
# 最重要：pages/13 と同じ呼び出し
# ============================================================
current_user, payload = get_current_user_from_session_or_cookie(st)

if current_user:
    st.success(f"✅ ログイン中: **{current_user}**")
else:
    st.warning("⚠️ 未ログイン（ポータルでログイン後に再読み込みしてください）")

st.divider()

# ============================================================
# 診断情報（bot_app 側）
# ============================================================
COOKIE_NAME = getattr(auth_cfg, "COOKIE_NAME", "prec_sso") if _auth_cfg_ok else "prec_sso"

cookie_present, cookie_preview = _debug_cookie_present(COOKIE_NAME)

cfg_file = getattr(auth_cfg, "__file__", None) if _auth_cfg_ok else None
jwt_file = getattr(jwt_utils, "__file__", None) if _jwt_ok else None

# JWT_SECRET の “型” と “中身の先頭” を出す（文字列でないと即死する）
if _auth_cfg_ok:
    jwt_secret = getattr(auth_cfg, "JWT_SECRET", None)
    jwt_secret_type = type(jwt_secret).__name__
    jwt_secret_preview = _safe_preview(jwt_secret, 8)
else:
    jwt_secret = None
    jwt_secret_type = None
    jwt_secret_preview = None

# env には入ってる？（SET/None だけ表示）
env_jwt_secret = "SET" if os.getenv("JWT_SECRET") else None

diag = {
    "THIS": str(_THIS),
    "PROJECTS_ROOT": str(PROJECTS_ROOT),

    # login state
    "current_user": current_user,
    "payload_present": bool(payload),
    "session_current_user": st.session_state.get("current_user"),

    # cookie
    "COOKIE_NAME": COOKIE_NAME,
    "extra_streamlit_components_available": _stx_ok,
    "cookie_present": cookie_present,
    "cookie_preview": cookie_preview,

    # import origin (これが違うと別物を読んでる)
    "auth_config_loaded": _auth_cfg_ok,
    "auth_config_error": _auth_cfg_err,
    "auth_config_file": cfg_file,

    "jwt_utils_loaded": _jwt_ok,
    "jwt_utils_error": _jwt_err,
    "jwt_utils_file": jwt_file,

    # secret situation
    "env_JWT_SECRET": env_jwt_secret,          # ← ここが minutes と違うなら「同じ条件」ではない
    "JWT_SECRET_type": jwt_secret_type,        # str 以外なら即アウト
    "JWT_SECRET_preview": jwt_secret_preview,  # dev-secr... か、portal と同じ値の先頭か

    # request meta
    "headers_host": st.context.headers.get("host") if hasattr(st, "context") else None,
    "headers_origin": st.context.headers.get("origin") if hasattr(st, "context") else None,
    "base_url": st.context.url if hasattr(st, "context") else None,
}

st.subheader("🔍 診断情報（bot_app 側）")
st.code(json.dumps(diag, ensure_ascii=False, indent=2), language="json")

st.markdown(
    """
**【見方（重要ポイントは3つだけ）】**

1. `auth_config_file` と `jwt_utils_file`  
   → minutes と **同じファイル**を読んでいるか（ここが違うと別物です）

2. `env_JWT_SECRET`  
   → minutes が `None` で bot が `SET` なら、**bot のプロセスだけ秘密鍵を持っています**

3. `JWT_SECRET_preview`  
   → minutes が `dev-secr...` で bot が別なら、**鍵が違うので minutes では verify できません**
"""
)

st.divider()
st.caption("このページは診断専用です（common_lib は変更しません）。")
