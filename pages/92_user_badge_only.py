# --- add this at the very top ---
from pathlib import Path
import sys

_THIS = Path(__file__).resolve()
# 親を順に見て、common_lib/COMMON_LIB があれば その親ディレクトリを sys.path に追加
for parent in [_THIS.parent, *_THIS.parents]:
    for name in ("common_lib", "COMMON_LIB"):
        if (parent / name).is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
                
# --- then your imports ---
from common_lib.ui.header_user_badge import get_current_user_from_cookie
import streamlit as st

user = get_current_user_from_cookie()
st.write("👤 ユーザー:", user or "未ログイン")