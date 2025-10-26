import streamlit as st
from lib.ui import hide_deploy_button
from config.path_config import PATHS  # ← 追加

st.set_page_config(page_title="社内ボット", page_icon="🤖", layout="wide")
hide_deploy_button()

st.title("🤖 社内ボット")
st.markdown("""
左の **Pages** から   
- **ボット**：保存した知識ベースに対して質問をします．
""")

st.info("『ボット』を使ってください。右側のサイドメニュー（『ボット』）をクリックしてください．")
st.error("このアプリは開発中です．『ボット』と『ポータルへ戻る』以外は使わないようにお願いします．")

# === ここから追加 ===
st.divider()
st.subheader("📂 現在の環境設定")

st.text(f"現在の location: {PATHS.preset}")
st.text(f"APP_ROOT       : {PATHS.app_root}（アプリフォルダーへのパス）")
st.text(f"pdf_root       : {PATHS.pdf_root}（ベクトル化するpdfファイルへのパス）")
st.text(f"backup_root    : {PATHS.backup_root}（データベースのバックアップ先（内部）へのパス）")
st.text(f"backup_root2    : {PATHS.backup_root2}（データベースのバックアップ先（外付け）へのパス）")
st.text(f"vs_root        : {PATHS.vs_root}（ベクトルデータベースへのパス）")
st.text(f"ssd_path       : {PATHS.ssd_path}（外付けSSDへのパス）")
# === ここまで追加 ===
