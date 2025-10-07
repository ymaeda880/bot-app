# ─────────────────────────────────────────────────────────────
# File: lib/pdf_ingest.py
# 目的: pages/05_pdfベクトル化.py からロジックを分離し、テスト容易性と再利用性を向上
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Set
import json

import pdfplumber
import numpy as np

# tiktoken は環境によって encoding_for_model が失敗することがあるのでフォールバック付きで準備
try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("text-embedding-3-large")
except Exception:
    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _enc = None  # 最終フォールバック（None のときは単純 len）

# 外部依存（本リポ内）
from lib.rag_utils import split_text, EmbeddingStore, NumpyVectorDB, ProcessedFilesSimple  # noqa: F401 (ProcessedFilesSimpleは他ファイルで使用)
from lib.vectorstore_utils import load_processed_files, save_processed_files
from lib.text_normalize import normalize_ja_text

OPENAI_EMBED_MODEL = "text-embedding-3-large"

# ============================================================
# Token helper
# ============================================================

def count_tokens(text: str) -> int:
    """テキストのトークン数を返す（tiktoken が無い/壊れている場合は文字数で代用）。"""
    if _enc is None:
        return len(text or "")
    try:
        return len(_enc.encode(text or ""))
    except Exception:
        return len(text or "")

# ============================================================
# パス列挙 / VS 生成
# ============================================================

def list_shards(pdf_root: Path) -> List[str]:
    """PDF_ROOT 直下のシャード（年度）一覧。"""
    if not pdf_root.exists():
        return []
    return sorted([p.name for p in pdf_root.iterdir() if p.is_dir()])


def list_pnos(pdf_root: Path, shard_id: str) -> List[str]:
    """指定 shard 配下の pno 一覧。"""
    d = pdf_root / shard_id
    if not d.exists():
        return []
    return sorted([p.name for p in d.iterdir() if p.is_dir()])


def list_pdfs_in_pno(pdf_root: Path, shard_id: str, pno: str) -> List[Path]:
    """`<pdf_root>/<shard>/<pno>` 直下の *.pdf / *.PDF を列挙（再帰なし）。"""
    d = pdf_root / shard_id / pno
    if not d.exists():
        return []
    files: List[Path] = []
    files.extend(sorted(d.glob("*.pdf")))
    files.extend(sorted(d.glob("*.PDF")))
    return files


def ensure_vs_dir(vs_root: Path, backend: str, shard_id: str) -> Path:
    """`<vs_root>/<backend>/<shard>` を作成して返す。"""
    d = vs_root / backend / shard_id
    d.mkdir(parents=True, exist_ok=True)
    return d

# ============================================================
# processed_files.json 関連（互換考慮）
# ============================================================

def _load_processed_keyset(pf_json_path: Path) -> Set[str]:
    """processed_files の代表形式を set[str] で返す（辞書/配列/混在を吸収）。"""
    done: Set[str] = set()
    try:
        with pf_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = None
    if not data:
        return done

    if isinstance(data, dict):
        vals = data.get("done") or data.get("files") or data.get("processed") or []
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, str):
                    done.add(v)
                elif isinstance(v, dict):
                    v2 = v.get("file") or v.get("path") or v.get("name")
                    if isinstance(v2, str):
                        done.add(v2)
        elif isinstance(vals, dict):
            for v in vals.values():
                if isinstance(v, str):
                    done.add(v)
        return done

    if isinstance(data, list):
        for e in data:
            if isinstance(e, str) and e:
                done.add(e)
            elif isinstance(e, dict):
                v = e.get("file") or e.get("path") or e.get("name")
                if isinstance(v, str) and v:
                    done.add(v)
    return done


def is_processed_any_form(done_set: Set[str], shard_id: str, pno: str, filename: str) -> bool:
    """現行/旧互換の 3 形式のいずれかに一致すれば処理済みと判断。"""
    key_full   = f"{shard_id}/{pno}/{filename}"
    key_shard  = f"{shard_id}/{filename}"
    key_legacy = filename
    return (key_full in done_set) or (key_shard in done_set) or (key_legacy in done_set)


def migrate_processed_files_to_canonical(pf_json: Path, shard_id: str, pno: str) -> None:
    """processed_files を `shard/pno/filename` に正規化。変更があれば保存。"""
    pf_list = load_processed_files(pf_json)
    if not pf_list:
        return
    changed = False
    canonical: List[str] = []

    for entry in pf_list:
        if isinstance(entry, str):
            val = entry
        elif isinstance(entry, dict):
            val = entry.get("file") or entry.get("path") or entry.get("name")
        else:
            continue
        if not val:
            continue

        parts = val.split("/")
        if len(parts) == 1:
            val = f"{shard_id}/{pno}/{val}"
            changed = True
        elif len(parts) == 2:
            if parts[0] == shard_id:
                val = f"{shard_id}/{pno}/{parts[1]}"
                changed = True
        canonical.append(val)

    seen, dedup = set(), []
    for v in canonical:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)

    if changed:
        save_processed_files(pf_json, dedup)

# ============================================================
# 除外ヘルパ（*_skip / _side.json: ocr ∈ {skipped, failed, locked}）
# ============================================================

_EXCLUDED_SIDE_OCR = {"skipped", "failed", "locked"}

def _is_skip_file(p: Path) -> bool:
    """ベース名に '_skip' を含む PDF を除外対象にする（大文字・小文字を区別しない）。"""
    return "_skip" in p.stem.lower()

def _is_side_excluded(p: Path) -> Tuple[bool, str]:
    """<basename>_side.json の ocr が {skipped, failed, locked} なら除外。"""
    side_path = p.with_name(p.stem + "_side.json")
    if not side_path.exists():
        return False, ""
    try:
        meta = json.loads(side_path.read_text(encoding="utf-8"))
    except Exception:
        return False, ""
    ocr_val = str(meta.get("ocr", "")).lower()
    if ocr_val in _EXCLUDED_SIDE_OCR:
        return True, f"side.json の ocr:'{ocr_val}'"
    return False, ""

def _filter_skip(paths: List[Path]) -> Tuple[List[Path], List[str]]:
    """*_skip と side.json(ocr∈{skipped,failed,locked}) を除外して (kept, logs) を返す。"""
    kept: List[Path] = []
    logs: List[str] = []
    for p in paths:
        if _is_skip_file(p):
            logs.append(f"不採用: {p.name} — `_skip` のため除外")
            continue
        is_ex, reason = _is_side_excluded(p)
        if is_ex:
            logs.append(f"不採用: {p.name} — {reason}")
            continue
        kept.append(p)
    return kept, logs

# ============================================================
# OCR 優先の候補決定
# ============================================================

def decide_ocr_candidates(pdf_paths: List[Path]) -> Tuple[List[Tuple[Path, str]], List[str]]:
    """同一ベース名に A.pdf / A_ocr.pdf がある場合は _ocr を採用してベースをスキップする。"""
    by_base: Dict[str, Dict[str, Path]] = {}
    for p in pdf_paths:
        stem = p.stem
        if stem.endswith("_ocr"):
            base = stem[:-4]
            by_base.setdefault(base, {})["ocr"] = p
        else:
            base = stem
            by_base.setdefault(base, {})["base"] = p

    candidates: List[Tuple[Path, str]] = []
    logs: List[str] = []

    for base, d in by_base.items():
        ocr = d.get("ocr")
        basep = d.get("base")
        if ocr is not None:
            candidates.append((ocr, "done"))
            if basep is not None:
                logs.append(f"🟢 採用: {ocr.name}（_ocr優先） / ⏭️ スキップ: {basep.name}")
            else:
                logs.append(f"🟢 採用: {ocr.name}（_ocr 単独）")
        elif basep is not None:
            candidates.append((basep, "no"))
            logs.append(f"🟢 採用: {basep.name}（_ocr なし）")

    candidates.sort(key=lambda t: t[0].name)
    return candidates, logs

# ============================================================
# 未処理プレビュー / ベクトル数
# ============================================================

def compute_unprocessed_map(pdf_root: Path, vs_root: Path, shard_id: str, backend: str) -> Dict[str, Dict[str, List[str] | str | int]]:
    """pno ごとの未処理状況（complete/none/partial と残件）を返す。"""
    result: Dict[str, Dict[str, List[str] | str | int]] = {}

    vs_dir = vs_root / backend / shard_id
    pf_json = vs_dir / "processed_files.json"
    done_set = _load_processed_keyset(pf_json) if pf_json.exists() else set()

    pnos_all = list_pnos(pdf_root, shard_id)

    for pno in pnos_all:
        raw_pdfs = list_pdfs_in_pno(pdf_root, shard_id, pno)

        # 除外（*_skip / side.json: ocr∈{skipped,failed,locked}）
        filtered_pdfs, _ = _filter_skip(raw_pdfs)

        # OCR優先の採否判定は filtered 後で実施
        candidates, _logs = decide_ocr_candidates(filtered_pdfs)
        cand_names = [p.name for (p, _ocr) in candidates]

        if not cand_names:
            result[pno] = {"status": "complete", "total_candidates": 0, "unprocessed_files": []}
            continue

        processed_flags = [is_processed_any_form(done_set, shard_id, pno, fn) for fn in cand_names]
        processed_count = sum(1 for x in processed_flags if x)

        if processed_count == 0:
            status = "none"; unproc: List[str] = cand_names[:]  # 全未処理
        elif processed_count == len(cand_names):
            status = "complete"; unproc = []
        else:
            status = "partial"; unproc = [fn for fn, ok in zip(cand_names, processed_flags) if not ok]

        result[pno] = {"status": status, "total_candidates": len(cand_names), "unprocessed_files": unproc}

    return result


def get_vector_count(base_dir: Path) -> int:
    """`vectors.npy` の行数（総ベクトル数）を返す（失敗時は 0）。"""
    p = base_dir / "vectors.npy"
    if not p.exists():
        return 0
    try:
        arr = np.load(p, mmap_mode="r")
        return int(arr.shape[0]) if arr.ndim == 2 else 0
    except Exception:
        return 0

# ============================================================
# 1 ファイル取り込み（抽出→分割→埋め込み→DB 追記）
# ============================================================

def ingest_pdf_file(
    vdb: NumpyVectorDB,
    estore: EmbeddingStore,
    shard_id: str,
    pno: str,
    pdf_path: Path,
    year_val: int | None,
    batch_size: int,
    chunk_size: int,
    overlap: int,
    ocr_flag: str,
    embed_model_label: str,
) -> Tuple[int, int]:
    """1 つの PDF を ingest して、(追加ファイル数, 追加チャンク数) を返す。

    追加ファイル数は通常 1 か 0（例外スキップ時）。
    """
    name = pdf_path.name
    new_files = 0
    new_chunks = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = max(len(pdf.pages), 1)

        for page_no, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
            raw = raw.replace("\t", " ").replace("\xa0", " ")
            text = " ".join(raw.split())
            if not text:
                continue

            text = normalize_ja_text(text)
            spans: List[Tuple[str, int, int]] = split_text(text, chunk_size=int(chunk_size), overlap=int(overlap))
            if not spans:
                continue

            chunks = [s[0] for s in spans]
            vectors: List[np.ndarray] = []
            metas: List[dict] = []

            for i in range(0, len(chunks), int(batch_size)):
                batch = chunks[i:i + int(batch_size)]
                vecs = estore.embed(batch, batch_size=int(batch_size)).astype("float32")
                vectors.append(vecs)

                for j, (ch, s, e) in enumerate(spans[i:i + int(batch_size)]):
                    metas.append({
                        "file": f"{shard_id}/{pno}/{name}",
                        "year": year_val,
                        "pno": pno,
                        "page": page_no,
                        "chunk_id": f"{name}#p{page_no}-{i+j}",
                        "chunk_index": i + j,
                        "text": ch,
                        "span_start": s,
                        "span_end": e,
                        "chunk_len_tokens": count_tokens(ch),
                        "ingested_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "shard_id": shard_id,
                        "embed_model": embed_model_label,
                        "ocr": "done" if ocr_flag == "done" else "no",
                    })

            vec_mat = np.vstack(vectors) if len(vectors) > 1 else vectors[0]
            vdb.add(vec_mat, metas)
            new_chunks += len(metas)

    new_files += 1
    return new_files, new_chunks
