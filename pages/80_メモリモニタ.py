# pages/80_メモリモニタ.py
from __future__ import annotations
import time
import streamlit as st
from lib.monitors.ui_memory_monitor import render_memory_monitor

st.set_page_config(page_title="🧠 メモリモニタ", page_icon="🧠", layout="wide")
st.title("🧠 メモリモニタ（プロセス & システム）")

# セッション管理
if "memmon_running" not in st.session_state:
    st.session_state.memmon_running = False
if "memmon_interval" not in st.session_state:
    st.session_state.memmon_interval = 3

# サイドバー
st.sidebar.header("⏱️ コントロール")
interval = st.sidebar.slider("更新間隔（秒）", 1, 10,
                             st.session_state.memmon_interval, key="memmon_interval")
col1, col2 = st.sidebar.columns(2)
if col1.button("▶️ 開始"):
    st.session_state.memmon_running = True
if col2.button("⏹ 停止"):
    st.session_state.memmon_running = False

# モニタUI呼び出し
render_memory_monitor(auto_update=st.session_state.memmon_running)


# ───────────────── 危険ラインの目安（折りたたみ表示）
with st.expander("⚠️ メモリ使用率の危険ライン（目安）", expanded=False):
    st.markdown("""
| 指標 | 安全域 | 注意域 | 危険域 | 説明 |
|------|---------|---------|---------|------|
| **全体RAM使用率（sys.percent）** | ～70% | 70〜85% | 85%以上 | OS全体の物理メモリ消費。80%超でスワップ開始の恐れ |
| **このアプリの占有率（proc.pct_of_system）** | ～30% | 30〜40% | 40%以上 | Pythonプロセスがメモリを専有。長時間の高止まりは危険 |
| **スワップ使用率（swap.percent）** | 0% | ～10% | 10%以上 | RAMが不足しディスクへ退避開始。性能低下のサイン |
| **メモリ圧力（pressure_hint）** | 0.0〜0.3 | 0.3〜0.6 | 0.6以上 | macOS専用。逼迫の近似指標。0.8以上で危険状態 |

🧭 **見方のポイント**
- 「全体RAM使用率」が 80% を超えるとシステム全体が重くなる。
- 「このアプリの占有率」が 40% 超ならRAG処理が肥大化している可能性。
- 「スワップ使用中」や「圧力ヒント ≥ 0.6」が出たら即GC・再起動検討。
- 「一時的な上昇」は正常だが、「高止まりが続く」場合はリークやキャッシュ過多を疑う。

💡 **おすすめ監視（現状未実装，実装予定）**
- RSS とスワップをセットで確認。
- `gc.collect()` 実行後に RSS が減らない場合 → NumPy / mmap の固定メモリ。
- 定期的に `max_loaded_shards` を制限して、不要シャードをアンロード。
    """)


# 自動更新
if st.session_state.memmon_running:
    time.sleep(st.session_state.memmon_interval)
    st.rerun()
