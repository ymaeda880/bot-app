# pages/52_シャード健全性チェック.py
# ------------------------------------------------------------
# 🧩 シャード健全性チェック（容量・件数・整合性）
# - PATHS.vs_root/<vectorspace>/<shard_id>/ を走査
# - vectors.npy の (行数n, 次元d), ファイルサイズ, meta.jsonl行数 を取得
# - しきい値で OK / WARN / NG を判定（理由つき）
# - RAM目安から「推奨最大ベクトル数/シャード」を自動試算（可編集）
# - 合計サイズは B/KB/MB/GB で人間向け表示（少量でも 0.00GB にならない）
# ------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
import sys

_THIS = Path(__file__).resolve()
APP_ROOT = _THIS.parents[1]          # pages -> app root
PROJECTS_ROOT = _THIS.parents[3]     # auth_portal/pages -> projects/auth_portal
import sys
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from typing import Dict, Any, List, Tuple
import math

import numpy as np
import streamlit as st

from config.path_config import PATHS  # 固定パス設定に一本化

from common_lib.auth.auth_helpers import require_admin_user

# ========== ページ設定 ==========
st.set_page_config(page_title="Shard Health Check", page_icon="🧩", layout="wide")

sub = require_admin_user(st)
if not sub:
    st.error("🚫 このページは管理者のみアクセスできます。")
    st.stop()

st.success(f"✅ 管理者ログイン中: **{sub}**")  # ← 表示はここで自由に

# ========== パス（PATHS から取得） ==========
VS_ROOT: Path = PATHS.vs_root  # <project>/data/vectorstore/<vectorspace>/<shard_id>/

# ========== ユーティリティ ==========
def help_popover(title: str, content_md: str) -> None:
    """Streamlit 1.31+ では popover、それより古い版では expander を使う"""
    pop = getattr(st, "popover", None)
    if callable(pop):
        with st.popover(title):
            st.markdown(content_md)
    else:
        with st.expander(title):
            st.markdown(content_md)

def list_shard_dirs(vectorspace: str) -> List[Path]:
    base = VS_ROOT / vectorspace
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])

def sizeof_fmt(num: float) -> str:
    """人間向けのサイズ表記（B/KB/MB/GB/TB）"""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(num) < 1024.0:
            return f"{num:,.2f} {unit}"
        num /= 1024.0
    return f"{num:,.2f} EB"

def load_vector_shape(vec_path: Path) -> Tuple[int, int]:
    """mmapで shape だけ取得（RAMに載せない）"""
    arr = np.load(vec_path, mmap_mode="r")
    shape = tuple(arr.shape)
    if len(shape) == 1:
        # 想定外（(n,)）の場合は (n, 1) 扱い
        return (shape[0], 1)
    return (shape[0], shape[1])

def count_jsonl_lines(jsonl_path: Path) -> int:
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def estimate_ram_for_vectors(n: int, d: int, dtype_bytes: int = 4) -> float:
    """float32=4bytes 基準で RAM 消費見積り（GB）"""
    return n * d * dtype_bytes / (1024**3)

def judge_status(
    row: Dict[str, Any],
    max_vectors_per_shard: int,
    max_vectors_file_gb: float,
    mismatch_tol_pct: float,
    est_ram_limit_gb: float,
) -> Tuple[str, List[str]]:
    """
    返り値: (status, reasons[])
    - status: "OK" | "WARN" | "NG"
    """
    reasons: List[str] = []
    n = int(row["n_vectors"])
    d = int(row["dim"])
    size_gb = float(row["vectors_npy_gb"])
    meta = int(row["meta_rows"])
    mismatch = abs(n - meta)
    mismatch_pct = (mismatch / max(1, n)) * 100.0 if n > 0 else 0.0
    est_ram_gb = float(row["est_ram_gb"])

    # チェック1: 件数しきい値
    if n > max_vectors_per_shard * 1.5:
        reasons.append(f"ベクトル数が上限の150%超（{n:,} > {int(max_vectors_per_shard*1.5):,}）")
    elif n > max_vectors_per_shard:
        reasons.append(f"ベクトル数が上限超（{n:,} > {max_vectors_per_shard:,}）")

    # チェック2: vectors.npy の物理サイズ
    if size_gb > max_vectors_file_gb * 1.5:
        reasons.append(f"vectors.npy が上限の150%超（{size_gb:.2f}GB > {max_vectors_file_gb*1.5:.2f}GB）")
    elif size_gb > max_vectors_file_gb:
        reasons.append(f"vectors.npy が上限超（{size_gb:.2f}GB > {max_vectors_file_gb:.2f}GB）")

    # チェック3: n と meta.jsonl 行数の不整合
    if mismatch_pct > mismatch_tol_pct * 2:
        reasons.append(f"meta不整合が閾値の2倍超（{mismatch_pct:.1f}% > {mismatch_tol_pct*2:.1f}%）")
    elif mismatch_pct > mismatch_tol_pct:
        reasons.append(f"meta不整合が閾値超（{mismatch_pct:.1f}% > {mismatch_tol_pct:.1f}%）")

    # チェック4: RAM見積り（安全域超過）
    if est_ram_gb > est_ram_limit_gb * 1.5:
        reasons.append(f"RAM見積りが安全上限の150%超（{est_ram_gb:.2f}GB > {est_ram_limit_gb*1.5:.2f}GB）")
    elif est_ram_gb > est_ram_limit_gb:
        reasons.append(f"RAM見積りが安全上限超（{est_ram_gb:.2f}GB > {est_ram_limit_gb:.2f}GB）")

    # ステータス決定
    if any("150%超" in r for r in reasons):
        status = "NG"
    elif reasons:
        status = "WARN"
    else:
        status = "OK"
    return status, reasons

def plan_split(n: int, max_per_shard: int) -> Tuple[int, List[Tuple[int, int]]]:
    """必要分割数と各シャードの件数割当（ほぼ均等）"""
    if n <= max_per_shard:
        return 1, [(n, 0)]
    k = math.ceil(n / max_per_shard)
    base = n // k
    rem = n % k
    plan: List[Tuple[int, int]] = []
    for i in range(k):
        cnt = base + (1 if i < rem else 0)
        plan.append((cnt, i))
    return k, plan

# ========== タイトル ==========
st.title("🧩 シャード健全性チェック（容量・件数・整合性）")

# ========== サイドバー ==========
with st.sidebar:
    st.header("設定")

    # ============================
    # ★ DB切替（vectorspace）
    # ============================
    st.subheader("📚 対象DB（vectorstore）")

    vectorspace_label = st.radio(
        "対象",
        ["report（openai）", "規定集（openai_sample）"],
        index=0,
        help="report: data/vectorstore/openai / 規定集: data/vectorstore/openai_sample",
        key="sb_vectorspace_label",
    )

    VECTORSPACE_MAP = {
        "report（openai）": "openai",
        "規定集（openai_sample）": "openai_sample",
    }
    vectorspace = VECTORSPACE_MAP[vectorspace_label]
    st.caption(f"現在: `{vectorspace}`")

    st.divider()

    # メモリ安全域
    st.markdown("### メモリ安全域（推奨）")
    ram_gb = st.number_input(
        "搭載RAM (GB)",
        min_value=4.0,
        max_value=1024.0,
        value=32.0,
        step=1.0,
        help="物理RAM容量。Dockerや他プロセスも使うため、安全係数で余裕を見ます。",
    )
    safety = st.slider(
        "安全係数（使用上限の割合）",
        0.1,
        0.9,
        0.5,
        0.05,
        help="RAM見積の合計が『搭載RAM×安全係数』を超えるとWARN/NG判定になります。",
    )
    est_ram_limit_gb = ram_gb * safety
    st.caption(f"推奨『検索時に使ってよいベクトルRAM合計上限』 ≈ **{est_ram_limit_gb:.1f} GB**")

    # しきい値
    st.markdown("### シャードしきい値")
    max_vectors_file_gb = st.number_input(
        "vectors.npy 1ファイルの推奨上限 (GB)",
        min_value=0.1,
        max_value=64.0,
        value=2.0,
        step=0.1,
        help="大きすぎる単一ファイルは配布・転送・バックアップで不利、という運用目安。",
    )
    max_vectors_per_shard = st.number_input(
        "ベクトル数/シャードの推奨上限 (件)",
        min_value=10_000,
        max_value=5_000_000,
        value=300_000,
        step=10_000,
        help="同時ロードや再ベクトル化、バックアップを想定した運用上の閾値。",
    )
    mismatch_tol_pct = st.slider(
        "meta不整合の許容率(%)",
        0.0,
        20.0,
        2.0,
        0.5,
        help="vectors.npy の n（行数）と meta.jsonl の行数のズレがこの割合を超えるとWARN/NG。",
    )

    st.markdown("### 追加オプション")
    sample_rows = st.number_input(
        "確認サンプル抽出（行）",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
        help="0=抽出しない。meta.jsonl の冒頭を表示します。",
    )

    # 参考表示：解決済みパス
    st.markdown("### 📂 現在の解決パス（参考）")
    st.text_input("VS_ROOT", str(VS_ROOT), disabled=True)
    st.text_input("Vectorstore Dir", str(VS_ROOT / vectorspace), disabled=True)

st.caption(f"スキャン対象: **{VS_ROOT} / {vectorspace}**")

# ========== 入力検証 ==========
base_dir = VS_ROOT / vectorspace
if not base_dir.exists():
    st.error(f"ベクトルルートが見つかりません: {base_dir}")
    st.stop()

shards = list_shard_dirs(vectorspace)
if not shards:
    st.warning("シャードが見つかりません。まずはベクトル化を実施してください。")
    st.stop()

# ========== 収集 ==========
rows: List[Dict[str, Any]] = []
total_vectors = 0
total_ram_est_gb = 0.0
total_size_bytes = 0  # bytesで合計

for shp in shards:
    vec = shp / "vectors.npy"
    meta = shp / "meta.jsonl"
    if not vec.exists():
        continue

    try:
        n, d = load_vector_shape(vec)
    except Exception as e:
        st.warning(f"{shp.name}: vectors.npy の読み込みでエラー: {e}")
        n, d = 0, 0

    meta_rows = count_jsonl_lines(meta)
    size_bytes = vec.stat().st_size if vec.exists() else 0
    size_gb = size_bytes / (1024**3)

    est_ram_gb = estimate_ram_for_vectors(n, d, dtype_bytes=4)

    row = dict(
        shard_id=shp.name,
        n_vectors=n,
        dim=d,
        vectors_npy=size_bytes,
        vectors_npy_gb=size_gb,
        meta_rows=meta_rows,
        mismatch=n - meta_rows,
        est_ram_gb=est_ram_gb,
        path=str(shp),
    )

    status, reasons = judge_status(
        row,
        max_vectors_per_shard=int(max_vectors_per_shard),
        max_vectors_file_gb=float(max_vectors_file_gb),
        mismatch_tol_pct=float(mismatch_tol_pct),
        est_ram_limit_gb=float(est_ram_limit_gb),
    )
    row["status"] = status
    row["reasons"] = "; ".join(reasons) if reasons else ""

    rows.append(row)

    total_vectors += n
    total_ram_est_gb += est_ram_gb
    total_size_bytes += size_bytes

# ========== 現在のパス設定 ==========
st.subheader("📂 PathConfig によるパス設定")
st.text(f"APP_ROOT     : {PATHS.app_root}")
st.text(f"vs_root      : {PATHS.vs_root}")
st.text(f"backup_root  : {PATHS.backup_root}")
st.text(f"pdf_root     : {PATHS.pdf_root}")
st.text(f"ssd_path     : {PATHS.ssd_path}")

st.divider()

# ========== 表示 ==========
st.subheader("📊 シャード一覧")
if not rows:
    st.info("vectors.npy を含むシャードが見つかりませんでした。")
    st.stop()

# 危険度高→低で並べ替え
priority = {"NG": 0, "WARN": 1, "OK": 2}
rows_sorted = sorted(rows, key=lambda r: (priority.get(r["status"], 3), -int(r["n_vectors"])))

# テーブル上部の補助ヘルプ
hc1, hc2 = st.columns([1, 0.18])
with hc1:
    st.caption("列『RAM見積』は、シャード単体を float32 で展開した場合の概算メモリ量です。")
with hc2:
    help_popover(
        "RAM見積とは？",
        r"""
**RAM見積**：ベクトルをメモリに展開したときの概算（GB）。  
$$
RAM_{GB}=\frac{n\times d\times 4}{1024^3}
$$
（4 = float32 のバイト数）

**例**：300,000 × 768 ≒ 0.86 GB  
**比較対象**：サイドバーの『搭載RAM×安全係数』。
""",
    )

def status_icon(s: str) -> str:
    return {"OK": "🟢 OK", "WARN": "🟡 WARN", "NG": "🔴 NG"}.get(s, s)

table = []
for r in rows_sorted:
    table.append({
        "シャード": r["shard_id"],
        "状態": status_icon(r["status"]),
        "ベクトル数 n": f'{int(r["n_vectors"]):,}',
        "次元 d": int(r["dim"]),
        "vectors.npy": sizeof_fmt(float(r["vectors_npy"])) + f" ({float(r['vectors_npy_gb']):.2f} GB)",
        "meta.jsonl 行数": f'{int(r["meta_rows"]):,}',
        "不一致 (n - meta)": f'{int(r["mismatch"]):+,}',
        "RAM見積": f'{float(r["est_ram_gb"]):.2f} GB',
        "理由": r["reasons"],
        "パス": r["path"],
    })

st.dataframe(table)

# ===== 合計 / 概況 =====
st.markdown("### 合計 / 概況")
cols = st.columns(3)
with cols[0]:
    st.metric("総ベクトル数", f"{total_vectors:,}")
with cols[1]:
    st.metric("vectors.npy 合計サイズ", sizeof_fmt(float(total_size_bytes)))
with cols[2]:
    st.metric(
        "RAM見積（全シャード）",
        f"{total_ram_est_gb*1024:.2f} MB" if total_ram_est_gb < 1.0 else f"{total_ram_est_gb:.2f} GB",
    )

# ========== 詳細（任意） ==========
with st.expander("🧪 meta.jsonl の先頭サンプルを確認（任意）", expanded=False):
    if sample_rows > 0:
        for shp in shards:
            meta = shp / "meta.jsonl"
            if not meta.exists():
                continue
            st.markdown(f"**{shp.name} / meta.jsonl**")
            try:
                with meta.open("r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= int(sample_rows):
                            break
                        st.code(line.rstrip("\n"))
            except Exception as e:
                st.warning(f"{shp.name}: meta.jsonl 読み込みエラー: {e}")
    else:
        st.info("サンプル抽出は 0 行に設定されています。サイドバーで変更してください。")

# ========== 推奨分割プラン ==========
st.subheader("🪓 大きすぎるシャードの分割プラン（目安）")
need_split = [r for r in rows_sorted if int(r["n_vectors"]) > int(max_vectors_per_shard)]
if not need_split:
    st.caption("現状のしきい値では、分割推奨のシャードはありません。")
else:
    for r in need_split:
        n = int(r["n_vectors"])
        k, plan = plan_split(n, int(max_vectors_per_shard))
        st.markdown(f"**{r['shard_id']}** は **{n:,} 件** → 推奨分割数 **{k}**")
        st.caption("割当（概算）: " + ", ".join([f"shard_{i}: {cnt:,}" for (cnt, i) in plan]))
        with st.expander(f"分割の運用ヒント: {r['shard_id']}", expanded=False):
            st.markdown(
                "- 元データの意味のある境界（年別・ファイル群など）で再ベクトル化すると運用が安定します。\n"
                f"- 生成時に new_shard_id を振り直し、`{str(VS_ROOT)}/{vectorspace}/<new_shard_id>/` に配置してください。\n"
                "- 既存の横断検索ページは複数 shard_id を選択可能なので、新構成でも運用できます。"
            )
