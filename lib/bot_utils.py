# lib/bot_utils.py
# =============================================================================
# RAG ボット向けユーティリティ（パス正規化 / シャード列挙 / 出典タグ拡張 ほか）
# （…既存ヘッダ略。ユーザーさんの現行ファイルをベースにしています…）
# =============================================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional, Iterable
import os
import re
import unicodedata
import json

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None

__all__ = [
    "list_shard_dirs_openai",
    "norm_path",
    "fmt_source",
    "citation_tag",
    "enrich_citations",
    "get_openai_api_key",
    "parse_inline_files",
    "strip_inline_files",
    # 追加分
    "to_halfwidth_digits",
    "clean_pno_token",
    "parse_years",
    "parse_pnos",
    "norm_pno_forms",
    "year_ok",
    "pno_ok",
    "file_ok",
    "scan_candidate_files",
    "filters_caption",
]

_INLINE_FILES_RE = re.compile(
    r"\[\[\s*files\s*:\s*([^\]]+)\]\]",
    flags=re.IGNORECASE,
)

# -------------------- 既存関数（そのまま） --------------------
def list_shard_dirs_openai(vs_root: Path) -> List[Path]:
    base = Path(vs_root) / "openai"
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])

def norm_path(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().replace("\\", "/")
    return s.lower()

def fmt_source(meta: Dict[str, Any]) -> str:
    f = str(meta.get("file", "") or "")
    p = meta.get("page", None)
    cid = str(meta.get("chunk_id", "") or "")
    if f and p is not None:
        try:
            base = f"{f} p.{int(p)}"
        except Exception:
            base = f
    else:
        base = f or "(unknown)"
    return f"{base} ({cid})" if cid else base

def citation_tag(i: int, meta: Dict[str, Any]) -> str:
    year = meta.get("year", None)
    pno = meta.get("pno", None) or meta.get("project_no", None)
    page = meta.get("page", None)
    parts: List[str] = [f"S{i}"]
    if year is not None and str(year).strip():
        parts.append(str(year))
    if pno is not None and str(pno).strip():
        parts.append(f"P{pno}")
    if page is not None:
        try:
            parts.append(f"p.{int(page)}")
        except Exception:
            pass
    return "[" + "|".join(parts) + "]"

def enrich_citations(text: str, raw_hits: List[Tuple[int, float, Dict[str, Any]]]) -> str:
    pattern = re.compile(r"\[S(\d+)(?:[^\]]*)\]")
    def _basename(path: str) -> str:
        if not path:
            return ""
        return path.split("/")[-1].split("\\")[-1]
    def _repl(m: re.Match) -> str:
        try:
            idx = int(m.group(1))
            if 1 <= idx <= len(raw_hits):
                _rid, _score, meta = raw_hits[idx - 1]
                year = meta.get("year", None)
                pno  = meta.get("pno", None) or meta.get("project_no", None)
                page = meta.get("page", None)
                file = meta.get("file", None)
                parts: List[str] = [f"S{idx}"]
                if year is not None and str(year).strip():
                    parts.append(str(year))
                if pno is not None and str(pno).strip():
                    parts.append(f"P{pno}")
                if page is not None:
                    try:
                        parts.append(f"p.{int(page)}")
                    except Exception:
                        pass
                if file:
                    fb = _basename(str(file))
                    if fb:
                        parts.append(fb)
                return "[" + "|".join(parts) + "]"
        except Exception:
            pass
        return m.group(0)
    return pattern.sub(_repl, text)

def get_openai_api_key() -> Optional[str]:
    try:
        if st is not None:
            ok = st.secrets.get("openai", {}).get("api_key")
            if ok:
                return str(ok)
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

def parse_inline_files(q: str) -> Set[str]:
    m = _INLINE_FILES_RE.search(q or "")
    if not m:
        return set()
    return {s.strip() for s in m.group(1).split(",") if s.strip()}

def strip_inline_files(q: str) -> str:
    return _INLINE_FILES_RE.sub("", q or "").strip()

# ==================== ここから追加ユーティリティ ====================

# ---- 入力正規化（normalization） -----------------------------------
def to_halfwidth_digits(s: str) -> str:
    """全角→半角の数字だけ変換（例: '２０２４'→'2024'）。"""
    return (s or "").translate(str.maketrans("０１２３４５６７８９", "0123456789"))

def clean_pno_token(t: str) -> str:
    """pno 入力からクォート/空白/記号を除去し、数字のみ返す。"""
    s = to_halfwidth_digits(t.strip().strip(" '\"“”’「」『』"))
    return "".join(ch for ch in s if ch.isdigit())

def parse_years(s: str) -> Set[int]:
    """'2019,2023' → {2019, 2023}（4桁だけ採用）。"""
    out: Set[int] = set()
    for tok in (s or "").split(","):
        digits = "".join(ch for ch in to_halfwidth_digits(tok).strip() if ch.isdigit())
        if len(digits) == 4:
            try:
                out.add(int(digits))
            except Exception:
                pass
    return out

def parse_pnos(s: str) -> Set[str]:
    """'010, 45, 120' → {'010','45','120'}（後段で表記ゆれを吸収）。"""
    out: Set[str] = set()
    for tok in (s or "").split(","):
        d = clean_pno_token(tok)
        if d:
            out.add(d)
    return out

def norm_pno_forms(s: str) -> Set[str]:
    """
    '010' → {'010','10','010'}
    '10'  → {'10','10','010'}
    空→ set()
    """
    s = clean_pno_token(s)
    if not s:
        return set()
    nozero = s.lstrip("0") or "0"
    return {s, nozero, nozero.zfill(3)}

# ---- メタ判定（year / pno / file） ---------------------------------
def _parse_year_value(v: Any) -> Optional[int]:
    y_raw = str(v or "")
    y_tok = "".join(ch for ch in to_halfwidth_digits(y_raw) if ch.isdigit())
    try:
        return int(y_tok) if len(y_tok) == 4 else None
    except Exception:
        return None

def year_ok(meta: Dict[str, Any], years: Set[int]) -> bool:
    """years が空なら常に True。非空なら meta['year'] が集合に含まれるか。"""
    if not years:
        return True
    y_val = _parse_year_value(meta.get("year"))
    return (y_val is not None) and (y_val in years)

def _pno_forms_from_meta(meta: Dict[str, Any]) -> Set[str]:
    p_raw = str(meta.get("pno") or meta.get("project_no") or "")
    d = to_halfwidth_digits(p_raw)
    digits = "".join(ch for ch in d if ch.isdigit())
    if not digits:
        return set()
    nozero = digits.lstrip("0") or "0"
    return {digits, nozero, nozero.zfill(3)}

def pno_ok(meta: Dict[str, Any], pnos: Set[str]) -> bool:
    """pnos が空なら常に True。非空なら pno の表記ゆれ集合が交差するか。"""
    if not pnos:
        return True
    forms = _pno_forms_from_meta(meta)
    return bool(forms and not forms.isdisjoint(pnos))

def file_ok(meta: Dict[str, Any], whitelist_norm: Set[str]) -> bool:
    """ホワイトリストが空なら True。非空なら meta['file'] が一致するか。"""
    if not whitelist_norm:
        return True
    return norm_path(str(meta.get("file", ""))) in whitelist_norm

# ---- meta.jsonl を軽く走査して候補ファイルを抽出 -------------------
# lib.vectorstore_utils.iter_jsonl が無くても動くようフォールバック
try:
    from lib.rag.vectorstore_utils import iter_jsonl as _iter_jsonl  # type: ignore
except Exception:
    def _iter_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
        if not p.exists():
            return []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def scan_candidate_files(
    shard_dirs: List[Path],
    years: Set[int],
    pnos: Set[str],
) -> Tuple[Dict[str, Set[str]], int]:
    """
    各シャードの meta.jsonl を“軽く”走査（quick scan）し、year/pno に合う
    file 候補を抽出する。テキスト本体は読まず、1行=1メタで判定のみ実施。
    """
    cand_by_shard: Dict[str, Set[str]] = {}
    total = 0
    if not (years or pnos):
        return cand_by_shard, total  # フィルタ無しならスキップ
    for sd in shard_dirs:
        meta_path = sd / "meta.jsonl"
        bucket: Set[str] = set()
        for m in _iter_jsonl(meta_path):
            if not year_ok(m, years):
                continue
            if not pno_ok(m, pnos):
                continue
            f = m.get("file")
            if f:
                bucket.add(norm_path(str(f)))
        if bucket:
            cand_by_shard[sd.name] = bucket
            total += len(bucket)
    return cand_by_shard, total

# ---- 画面表示用のキャプション -------------------------------------
def filters_caption(years: Set[int], pnos: Set[str], cand_total: Optional[int]) -> str:
    ys = sorted(list(years)) or "（未指定）"
    ps = sorted(list(pnos)) or "（未指定）"
    tail = "（スキップ）" if cand_total is None else str(cand_total)
    return f"適用フィルタ → year: {ys} / pno: {ps} / 候補ファイル数: {tail}"
