# lib/processed_files_utils.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Any
import json, os, urllib.parse, unicodedata

def _canon(s: str) -> str:
    if not s:
        return ""
    s = urllib.parse.unquote(s)
    s = unicodedata.normalize("NFKC", s).strip()
    s = s.replace("\\", "/")
    s = os.path.normpath(s).replace("\\", "/")
    return s.lower()

def _entry_to_pathlike(entry) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for k in ("file", "path", "name", "relpath", "source", "original", "orig", "pdf"):
            v = entry.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""

def _load_pf_struct(pf_path: Path):
    if not pf_path.exists():
        return "empty", None, []
    try:
        root = json.loads(pf_path.read_text(encoding="utf-8"))
    except Exception:
        return "unknown", None, []

    if isinstance(root, dict) and isinstance(root.get("done"), list):
        return "object_done", root, [root["done"]]

    if isinstance(root, list):
        done_lists = []
        all_str = True
        for e in root:
            if isinstance(e, dict) and isinstance(e.get("done"), list):
                done_lists.append(e["done"])
                all_str = False
            elif not isinstance(e, str):
                all_str = False
        if done_lists:
            return "array_of_done_objects", root, done_lists
        if all_str:
            return "array", root, [root]

    return "unknown", root, []

def _save_pf_struct(pf_path: Path, schema: str, root_obj):
    if schema in ("object_done", "array", "array_of_done_objects") and root_obj is not None:
        pf_path.write_text(json.dumps(root_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    items: list[str] = []
    if isinstance(root_obj, dict) and isinstance(root_obj.get("done"), list):
        for e in root_obj["done"]:
            s = _entry_to_pathlike(e)
            if s:
                items.append(s)
    elif isinstance(root_obj, list):
        for e in root_obj:
            if isinstance(e, dict) and isinstance(e.get("done"), list):
                for x in e["done"]:
                    s = _entry_to_pathlike(x)
                    if s:
                        items.append(s)
            else:
                s = _entry_to_pathlike(e)
                if s:
                    items.append(s)
    items = sorted(set(items))
    pf_path.write_text(json.dumps({"done": items}, ensure_ascii=False, indent=2), encoding="utf-8")

def remove_from_processed_files_selective(pf_path: Path, removed_files: List[str]) -> Tuple[int, int, int, List[str]]:
    """
    processed_files.json の構造を維持したまま、removed_files に該当する項目を取り除く。
    戻り値: (before_total, after_total, removed_count, removed_examples[:10])
    """
    schema, root, list_refs = _load_pf_struct(pf_path)
    if not list_refs:
        return (0, 0, 0, [])

    t_full = {_canon(x) for x in removed_files}
    t_base = {os.path.basename(x) for x in t_full}
    t_stem = {os.path.splitext(b)[0] for b in t_base}

    def _match(entry) -> bool:
        raw = _entry_to_pathlike(entry)
        cn = _canon(raw)
        if not cn:
            return False
        base = os.path.basename(cn)
        stem = os.path.splitext(base)[0]
        return (
            (cn in t_full) or
            (base in t_base) or
            (stem in t_stem) or
            any(cn.endswith("/" + t) for t in t_full)
        )

    before_total = sum(len(lst) for lst in list_refs)
    removed_show: list[str] = []

    for lst in list_refs:
        new_lst = []
        for e in lst:
            if _match(e):
                raw = _entry_to_pathlike(e)
                removed_show.append(raw if raw else json.dumps(e, ensure_ascii=False)[:120])
            else:
                new_lst.append(e)
        lst.clear()
        lst.extend(new_lst)

    _save_pf_struct(pf_path, schema, root)

    after_total = sum(len(lst) for lst in list_refs)
    removed_count = before_total - after_total
    return (before_total, after_total, removed_count, removed_show[:10])
