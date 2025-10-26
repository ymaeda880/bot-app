# pages/91_cookie_debug.py  — 右上ユーザーバッジが必ず出る版
from __future__ import annotations
from pathlib import Path
import sys, json, datetime as dt, base64, jwt
import streamlit as st

# ---------- 0) 最初に page_config ----------
st.set_page_config(page_title="Cookie Debug（SSO検証）", page_icon="🍪", layout="wide")

# ---------- 1) common_lib パス ----------
def _add_commonlib_to_syspath() -> str | None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        for name in ("common_lib", "COMMON_LIB"):
            cand = parent / name
            if cand.is_dir():
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                return str(cand)
    return None
_commonlib_dir = _add_commonlib_to_syspath()

# ---------- 2) 依存 ----------
try:
    import extra_streamlit_components as stx
except Exception:
    st.error("extra-streamlit-components が必要です: pip install extra-streamlit-components")
    st.stop()

# ---------- 3) SSO 設定 & verify ----------
COOKIE_NAME = "prec_sso"
verify_jwt = None
jwt_constants = {}
try:
    from common_lib.auth.config import COOKIE_NAME as _CN, JWT_SECRET, JWT_ALGO, JWT_AUD, JWT_ISS
    COOKIE_NAME = _CN or COOKIE_NAME
    jwt_constants = dict(JWT_SECRET=JWT_SECRET, JWT_ALGO=JWT_ALGO, JWT_AUD=JWT_AUD, JWT_ISS=JWT_ISS)
    from common_lib.auth.jwt_utils import verify_jwt as _verify
    verify_jwt = _verify
except Exception:
    pass

# ---------- 4) CookieManager は1回だけ作る ----------
cm = stx.CookieManager(key="cm-cookie-debug")

# ★ ここが重要：最初に一度 get_all() でマウント＆同期させる
cookies = cm.get_all() or {}

# ---------- 5) 右上ユーザーバッジ（common_lib ヘルパー利用） ----------
try:
    from common_lib.ui.header_user_badge import render_user_badge_right
    # ヘルパーは cm.get() を使う実装だが、初回空回りを避けたいので
    # ここで token を優先取得（なければヘルパー内部の取得にフォールバック）
    token = cookies.get(COOKIE_NAME)
    if token:
        # 署名検証 → sub を取り出し
        user = None
        try:
            payload = verify_jwt(token) if verify_jwt else None
            user = (payload or {}).get("sub") if payload else None
        except Exception:
            user = None
        if not user:
            # no-verify デコードで sub を拾う
            try:
                payload_b64 = token.split(".")[1] + "==="
                user = json.loads(base64.urlsafe_b64decode(payload_b64)).get("sub")
            except Exception:
                user = None
        # 直接描画（ヘルパーに任せてもOK）
        badge_html = (f"<div style='position:fixed; top:10px; right:20px; background:#eef9ee; color:#222;"
                      f"padding:6px 12px; border-radius:12px; font-weight:600;"
                      f"box-shadow:0 1px 3px rgba(0,0,0,0.1); z-index:99999;'>👤 {user}</div>"
                      if user else
                      "<div style='position:fixed; top:10px; right:20px; background:#f4f4f4; color:#888;"
                      "padding:6px 12px; border-radius:12px; font-weight:500;"
                      "box-shadow:0 1px 2px rgba(0,0,0,0.05); z-index:99999;'>未ログイン</div>")
        st.markdown(badge_html, unsafe_allow_html=True)
    else:
        # ヘルパーに任せる（内部で cm.get を呼ぶ）
        render_user_badge_right(cm=cm, cookie_name=COOKIE_NAME)
except Exception:
    # ヘルパーが無い場合の最低限表示
    st.markdown(
        "<div style='position:fixed; top:10px; right:20px; background:#f4f4f4; color:#888;"
        "padding:6px 12px; border-radius:12px; font-weight:500; "
        "box-shadow:0 1px 2px rgba(0,0,0,0.05); z-index:99999;'>未ログイン</div>",
        unsafe_allow_html=True
    )

# ---------- 6) 以降は通常の UI ----------
st.title("🍪 Cookie Debug（SSO / JWT 検証）")
st.caption(f"探す SSO Cookie 名: **{COOKIE_NAME}** / common_lib: {_commonlib_dir or '(not found)'}")

st.subheader("1) ブラウザから送信された Cookie 一覧")
if not cookies:
    st.info("受信クッキーがありません。別タブでログインしてからリロードしてください。")
else:
    def _short(v: str, n=60): return v if len(v) <= n else v[:28] + " … " + v[-28:]
    rows = [{"name": k, "length": len(str(v)), "value_preview": _short(str(v))} for k, v in cookies.items()]
    st.dataframe(rows, use_container_width=True)

st.subheader("2) SSO クッキー（JWT）検査")
token = cookies.get(COOKIE_NAME)
if not token:
    st.warning(f"SSO Cookie **{COOKIE_NAME}** が見つかりません。")
    st.stop()

st.success(f"SSO Cookie {COOKIE_NAME} を検出（長さ {len(token)}）。")
with st.expander("値のプレビュー（先頭/末尾のみ）", expanded=False):
    st.code(f"{token[:100]} … {token[-40:]}")

# 検証
payload, err = None, None
if verify_jwt:
    try:
        payload = verify_jwt(token)
    except Exception as e:
        err = f"verify_jwt 例外: {e}"

if payload:
    st.markdown("✅ **JWT 検証 OK**")
    st.json(payload)
else:
    st.error("JWT 検証に失敗 or verify_jwt が見つかりません。")
    if err: st.caption(err)
    try:
        st.info("署名検証なしのデコード結果：")
        st.json(jwt.decode(token, options={"verify_signature": False}))
    except Exception as e:
        st.warning(f"decode(no-verify) 失敗: {e}")

# 期限表示
if payload:
    jst = dt.timezone(dt.timedelta(hours=9), name="Asia/Tokyo")
    def _fmt_ts(v):
        try: return dt.datetime.fromtimestamp(int(v), tz=jst).strftime("%Y-%m-%d %H:%M:%S %Z")
        except: return str(v)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("sub", str(payload.get("sub") or payload.get("user") or "—"))
    with c2: st.metric("iat", _fmt_ts(payload.get("iat")))
    with c3: st.metric("nbf", _fmt_ts(payload.get("nbf")))
    with c4: st.metric("exp", _fmt_ts(payload.get("exp")))
