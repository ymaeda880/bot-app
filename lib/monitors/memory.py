# lib/monitors/memory.py
from __future__ import annotations
import os, gc, subprocess, shlex
from dataclasses import dataclass
from typing import Optional
import psutil

@dataclass
class ProcMem:
    rss_bytes: int       # 実メモリ（常駐集合）
    vms_bytes: int       # 仮想メモリサイズ
    pct_of_system: float # 全体RAMに対するパーセント

@dataclass
class SysMem:
    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent: float       # 使用率(%)

@dataclass
class SwapMem:
    total_bytes: int
    used_bytes: int
    percent: float

@dataclass
class MemSnapshot:
    proc: ProcMem
    sys: SysMem
    swap: SwapMem
    pressure_hint: Optional[float]  # macOSの簡易圧力(0〜1想定) or None

def _macos_memory_pressure_hint() -> Optional[float]:
    """
    macOS向けのざっくりヒント。`vm_stat` を読み取り、
    free+inactive を total で割って逆数っぽく0..1正規化（かなり簡易）。
    """
    try:
        out = subprocess.check_output(shlex.split("vm_stat"), text=True)
        page_size = 4096
        pages = {}
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = int(v.strip().strip(".").replace(".", "").replace(",", "").split()[0])
                pages[k] = v
        total_like = sum(pages.get(k, 0) for k in ["Pages active", "Pages inactive", "Pages speculative", "Pages wired down", "Pages free"])
        free_like  = pages.get("Pages free", 0) + pages.get("Pages inactive", 0)
        if total_like <= 0:
            return None
        free_ratio = free_like / total_like
        return max(0.0, min(1.0, 1.0 - free_ratio))  # 0(余裕)〜1(圧力大)
    except Exception:
        return None

def snapshot() -> MemSnapshot:
    p = psutil.Process(os.getpid())
    pmem = p.memory_info()
    vmem = psutil.virtual_memory()
    smem = psutil.swap_memory()
    pressure = _macos_memory_pressure_hint()

    proc = ProcMem(
        rss_bytes=pmem.rss,
        vms_bytes=pmem.vms,
        pct_of_system=(pmem.rss / vmem.total * 100.0) if vmem.total else 0.0,
    )
    sysm = SysMem(
        total_bytes=vmem.total,
        available_bytes=vmem.available,
        used_bytes=vmem.used,
        percent=vmem.percent,
    )
    swapm = SwapMem(
        total_bytes=smem.total,
        used_bytes=smem.used,
        percent=smem.percent,
    )
    return MemSnapshot(proc=proc, sys=sysm, swap=swapm, pressure_hint=pressure)

def humanize(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def force_gc():
    gc.collect()
