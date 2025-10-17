# lib/monitors/ui_memory_monitor.py
from __future__ import annotations
import time, datetime as dt
import streamlit as st
from lib.monitors.memory import snapshot, humanize, force_gc


# =============================================================================
# 💡 フルバージョン（モニタページ用）
# =============================================================================
def render_memory_monitor(auto_update: bool = True) -> None:
    """メモリモニタのUI全体を描画（他ページでも呼び出し可）"""
    snap = snapshot()
    now = dt.datetime.now().strftime("%H:%M:%S")

    # ────────────── スタイル
    st.markdown("""
    <style>
      .blink-dot {width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;background:#22c55e;animation:blink 1s infinite;}
      @keyframes blink {50%{opacity:.2;}}
      .status-pill {display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;background:#f1f5f9;font-size:.95rem;}
      .muted {color:#6b7280;font-size:.85rem;}
    </style>
    """, unsafe_allow_html=True)

    if auto_update:
        st.markdown(f'<div class="status-pill"><span class="blink-dot"></span>'
                    f'更新中<span class="muted"> 最終更新 {now}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-pill">⏸ 停止中<span class="muted"> 最終更新 {now}</span></div>',
                    unsafe_allow_html=True)

    st.write("")

    # ────────────── このアプリ
    st.subheader("🧠 このアプリ（Pythonプロセス）")
    c1, c2, c3 = st.columns(3)
    c1.write("**RSS（実メモリ）**")
    c1.write(humanize(snap.proc.rss_bytes))
    c1.caption("Resident Set Size: 実際にRAM上に常駐している量")

    c2.write("**VMS（仮想メモリ）**")
    c2.write(humanize(snap.proc.vms_bytes))
    c2.caption("プロセスが確保した仮想空間全体")

    c3.write("**全体RAMに占める割合**")
    c3.write(f"{snap.proc.pct_of_system:.1f}%")
    c3.caption("このプロセスが物理メモリに占める割合")

    st.divider()

    # ────────────── システム全体
    st.subheader("💻 システム全体")
    c4, c5, c6 = st.columns(3)
    c4.write("**物理メモリ合計**")
    c4.write(humanize(snap.sys.total_bytes))
    c4.caption("搭載RAMの合計容量")

    c5.write("**利用中（概算）**")
    c5.write(humanize(snap.sys.used_bytes))
    c5.caption("現在使用されている物理メモリ量")

    c6.write("**使用率**")
    c6.write(f"{snap.sys.percent:.1f}%")
    c6.caption("システム全体でのメモリ使用率")

    st.divider()
    colA, colB, colC = st.columns(3)
    colA.subheader("📦 スワップ")
    colA.write(f"**使用中:** {humanize(snap.swap.used_bytes)}")
    colA.write(f"**使用率:** {snap.swap.percent:.1f}%")
    colA.caption("RAM不足時にディスクへ退避された領域")

    colB.subheader("🌡️ 圧力ヒント（macOS）")
    if snap.pressure_hint is None:
        colB.write("取得不可")
    else:
        colB.progress(min(1.0, snap.pressure_hint),
                      text=f"pressure≈{snap.pressure_hint:.2f}（0余裕〜1逼迫）")

    colC.subheader("🧹 操作")
    if colC.button("GC（キャッシュ/未参照の解放）", key="memmon_gc_btn"):
        force_gc()
        st.success("gc.collect() 実行しました", icon="✅")

    # ────────────── サマリー
    st.divider()
    st.subheader("📋 メモリ状況サマリー")
    summary_data = [
        ["RSS（実メモリ）", humanize(snap.proc.rss_bytes), "RAM上に展開している量"],
        ["VMS（仮想メモリ）", humanize(snap.proc.vms_bytes), "確保された仮想空間全体"],
        ["RAM使用率", f"{snap.proc.pct_of_system:.1f}%", "プロセスの物理メモリ占有率"],
        ["システム使用率", f"{snap.sys.percent:.1f}%", "全体のメモリ利用率"],
        ["スワップ", humanize(snap.swap.used_bytes), f"{snap.swap.percent:.1f}%"],
        ["メモリ圧力", 
            "取得不可" if snap.pressure_hint is None else f"{snap.pressure_hint:.2f}",
            "macOSのみの逼迫指標"],
    ]
    st.table({"項目": [r[0] for r in summary_data],
              "値": [r[1] for r in summary_data],
              "説明": [r[2] for r in summary_data]})


# =============================================================================
# ⚡ 軽量バージョン（Botページ埋め込み用）
# =============================================================================
def render_memory_kpi_row() -> None:
    """Botページ下部などに埋め込む軽量1行KPI風メモリ状況"""
    snap = snapshot()
    now = dt.datetime.now().strftime("%H:%M:%S")

    st.markdown("""
    <style>
      .kpi-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;}
      .kpi {
        padding:8px 10px;border-radius:12px;background:#ffffff;border:1px solid #e5e7eb;
        box-shadow:0 1px 2px rgba(0,0,0,.05);text-align:center;
      }
      .kpi .label {font-size:.8rem;color:#6b7280;}
      .kpi .value {font-size:1rem;font-weight:600;}
      .muted {color:#94a3b8;font-size:.8rem;text-align:right;}
    </style>
    """, unsafe_allow_html=True)

    rss = humanize(snap.proc.rss_bytes)
    swap = humanize(snap.swap.used_bytes)
    pct = f"{snap.proc.pct_of_system:.1f}%"
    pressure = "取得不可" if snap.pressure_hint is None else f"{snap.pressure_hint:.2f}"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="label">🧠 RSS（実メモリ）</div><div class="value">{rss}</div></div>
      <div class="kpi"><div class="label">📦 スワップ</div><div class="value">{swap}</div></div>
      <div class="kpi"><div class="label">💻 RAM使用率</div><div class="value">{pct}</div></div>
    </div>
    <div class="muted">最終更新: {now}　圧力: {pressure}</div>
    """, unsafe_allow_html=True)
