#bot_app/app.py
#
import streamlit as st
from lib.ui import hide_deploy_button
from config.path_config import PATHS  # ← 追加

# ============================================================
# パスの取得とcommon_lib読み込み（app.pyにおけるコード）
# ============================================================
from pathlib import Path
import sys

_THIS = Path(__file__).resolve()
APP_ROOT = _THIS.parent
APP_NAME = APP_ROOT.name                  # ← app_name を自動取得
PROJECTS_ROOT = _THIS.parents[2]

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from common_lib.sessions import SessionConfig, init_session, heartbeat_tick
from common_lib.auth.auth_helpers import require_login
from common_lib.ui.banner_lines import render_banner_line_by_key


# ============================================================
# set_page_config
# ============================================================
st.set_page_config(page_title="Bot / 社内ボット", page_icon="🤖", layout="wide")
render_banner_line_by_key("cyan_clean")

# ============================================================
# Session heartbeat（全ページ共通・app.py）
# ============================================================
from common_lib.storage.storages_config import resolve_storages_root

STORAGES_ROOT = resolve_storages_root(PROJECTS_ROOT)

SESSIONS_DB = (
    STORAGES_ROOT
    / "_admin"
    / "sessions"
    / "sessions.db"
)

CFG = SessionConfig()  # heartbeat=30s, TTL=120s（既定）

# ───────────────── ログイン必須 ─────────────────

sub = require_login(st)
if not sub:
    st.stop()

# ───────────────── ヘッダ ─────────────────
left, right = st.columns([2, 1])
with left:
    st.title("🤖 社内ボット")
with right:
    st.success(f"✅ ログイン中: **{sub}**")

user = sub

# ───────────────── sessions（初期化 + heartbeat） ─────────────────
init_session(db_path=SESSIONS_DB, cfg=CFG, user_sub=user, app_name=APP_NAME)
heartbeat_tick(db_path=SESSIONS_DB, cfg=CFG, user_sub=user, app_name=APP_NAME)

#hide_deploy_button()

# st.title("🤖 Bot / 社内ボット")
# st.markdown("""
# 左の **Pages** から   
# - **ボット**：保存した知識ベースに対して質問をします．
# """)

st.info("『ボット』を使ってください。右側のサイドメニュー（『ボット』）をクリックしてください．")
st.caption("このアプリは開発中です．『ボット』と『ポータルへ戻る』以外は使わないようにお願いします．")

# === ここから追加 ===
st.divider()

st.markdown("""
## 📝 社内ボットアプリケーションについて（開発中のお知らせ）

本アプリケーションが参照する報告書データについては，**2019年分から2020年まで（データベース上に報告書が存在するもの）と、2021年の一部** が現在データベース化されています。
今後は、データベースを **順次拡充** していく予定です。（pdfファイルが400以上存在するプロジェクトは現在は除外しています．）

本アプリケーションは現在も開発を進めており、 使い勝手の面でご不便をおかけする部分があるかもしれませんが、 
皆様からの **積極的なご利用とフィードバック** が改善の大きな力となります。

より良い社内ボットアプリケーションを共に育てていくため、 何卒ご協力のほどよろしくお願いいたします。
""")



with st.expander("📂 現在の環境設定（クリックして開く）", expanded=False):
    st.text(f"現在の location : {PATHS.preset}")
    st.text(f"APP_ROOT       : {PATHS.app_root}（アプリフォルダーへのパス）")
    st.text(f"pdf_root       : {PATHS.pdf_root}（ベクトル化するPDFファイルへのパス）")
    st.text(f"backup_root    : {PATHS.backup_root}（データベースのバックアップ先（内部）へのパス）")
    st.text(f"backup_root2   : {PATHS.backup_root2}（データベースのバックアップ先（外付け）へのパス）")
    st.text(f"vs_root        : {PATHS.vs_root}（ベクトルデータベースへのパス）")
    st.text(f"ssd_path       : {PATHS.ssd_path}（外付けSSDへのパス）")
