# lib/bot/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set

import heapq
from itertools import count

from config.path_config import PATHS
from config.config import DEFAULT_USDJPY, estimate_tokens_from_text

from lib.text_normalize import normalize_ja_text
from lib.prompts.bot_prompt import build_prompt
from lib.gpt_responder import GPTResponder
from lib.gemini_responder import GeminiResponder
from lib.rag.rag_utils import EmbeddingStore, NumpyVectorDB
from lib.openai_utils import count_tokens
from lib.costs_new import (
    estimate_chat_cost,
    estimate_embedding_cost,
    ChatUsage,
)
from lib.bot_utils import (
    list_shard_dirs_openai,
    fmt_source,
    enrich_citations,
    year_ok,
    pno_ok,
    scan_candidate_files,
)


# ============================================================
# View Model
# pages / 履歴表示 / 再表示 すべてで共通に使う
# ============================================================

@dataclass
class BotAnswerView:
    answer_text: str
    raw_hits: List[Tuple[int, float, Dict[str, Any]]]

    # usage / cost
    embedding_tokens: int
    chat_input_tokens: int
    chat_output_tokens: int
    cost_usd: float
    cost_jpy: float

    # model info
    chat_model: str
    used_gemini: bool


# ============================================================
# 実行パイプライン（封印対象）
# ============================================================

def run_bot_query(
    *,
    question: str,
    chat_model: str,
    detail: str,
    max_tokens: int,
    top_k: int,
    years_sel: Set[int],
    pnos_sel_norm: Set[str],
    system_instruction: str,
) -> BotAnswerView:
    """
    1 回の質問処理を完結させる実行パイプライン。

    pages 側はこの関数だけを呼ぶ。
    検索・生成・usage・コスト計算の詳細はすべてここに封印する。
    """

    use_gemini = chat_model.startswith("gemini-")

    # --------------------------------------------------------
    # 1. shard 列挙
    # --------------------------------------------------------
    vs_backend_dir = PATHS.vs_root / "openai"
    shard_ids = [p.name for p in list_shard_dirs_openai(PATHS.vs_root)]
    shard_dirs = [
        vs_backend_dir / sid
        for sid in shard_ids
        if (vs_backend_dir / sid / "vectors.npy").exists()
    ]

    if not shard_dirs:
        raise RuntimeError("検索可能なシャードが存在しません")

    # --------------------------------------------------------
    # 2. 候補ファイル事前チェック（year / pno）
    # --------------------------------------------------------
    _, cand_total = scan_candidate_files(shard_dirs, years_sel, pnos_sel_norm)
    if (years_sel or pnos_sel_norm) and cand_total == 0:
        raise RuntimeError("指定の year / pno に一致する候補ファイルが 0 件です")

    # --------------------------------------------------------
    # 3. Embedding
    # --------------------------------------------------------
    norm_q = normalize_ja_text(question)
    estore = EmbeddingStore(backend="openai")

    embedding_tokens = count_tokens(question, "text-embedding-3-large")
    qv = estore.embed([norm_q]).astype("float32")

    # --------------------------------------------------------
    # 4. 検索（Top-K マージ）
    # --------------------------------------------------------
    heap: List[Tuple[float, int, int, Dict[str, Any]]] = []
    tie = count()

    for shp in shard_dirs:
        vdb = NumpyVectorDB(shp)
        local_k = max(top_k * 10, 50) if (years_sel or pnos_sel_norm) else top_k
        hits = vdb.search(qv, top_k=local_k, return_="similarity")

        for h in hits:
            if isinstance(h, tuple) and len(h) == 3:
                row_idx, score, meta = h
            else:
                score, meta = h
                row_idx = -1

            md = dict(meta or {})
            md["shard_id"] = shp.name

            if not year_ok(md, years_sel):
                continue
            if not pno_ok(md, pnos_sel_norm):
                continue

            sc = float(score)
            if len(heap) < top_k:
                heapq.heappush(heap, (sc, next(tie), row_idx, md))
            elif sc > heap[0][0]:
                heapq.heapreplace(heap, (sc, next(tie), row_idx, md))

    raw_hits = [
        (rid, sc, md)
        for (sc, _tb, rid, md) in sorted(heap, key=lambda x: x[0], reverse=True)
    ]

    if not raw_hits:
        raise RuntimeError("該当するコンテキストが見つかりませんでした")

    # --------------------------------------------------------
    # 5. プロンプト構築
    # --------------------------------------------------------
    labeled = [
        f"[S{i}] {m.get('text', '')}\n"
        f"[meta: {fmt_source(m)} / score={float(s):.3f}]"
        for i, (_rid, s, m) in enumerate(raw_hits, 1)
    ]

    prompt = build_prompt(
        norm_q,
        labeled,
        sys_inst=system_instruction,
        style_hint=detail,
        cite=True,
        strict=False,
    )

    # --------------------------------------------------------
    # 6. LLM 呼び出し
    # --------------------------------------------------------
    if use_gemini:
        responder = GeminiResponder()
        result = responder.complete(
            model=chat_model,
            system_instruction=system_instruction,
            user_content=prompt,
            max_output_tokens=max_tokens,
        )
        answer = result.text or ""
        chat_input_tokens = estimate_tokens_from_text(prompt)
        chat_output_tokens = estimate_tokens_from_text(answer)

    else:
        responder = GPTResponder()
        result = responder.complete(
            model=chat_model,
            system_instruction=system_instruction,
            user_content=prompt,
            max_output_tokens=max_tokens,
        )
        answer = result.text or ""
        chat_input_tokens = result.usage.input_tokens
        chat_output_tokens = result.usage.output_tokens

    # --------------------------------------------------------
    # 7. 出典整形
    # --------------------------------------------------------
    answer = enrich_citations(answer, raw_hits)

    # --------------------------------------------------------
    # 8. コスト計算
    # --------------------------------------------------------
    emb_cost = estimate_embedding_cost(
        "text-embedding-3-large",
        embedding_tokens,
        rate=DEFAULT_USDJPY,
    )

    chat_cost = estimate_chat_cost(
        chat_model,
        ChatUsage(
            input_tokens=int(chat_input_tokens),
            output_tokens=int(chat_output_tokens),
        ),
        rate=DEFAULT_USDJPY,
    )

    total_usd = float(emb_cost["usd"]) + float(chat_cost["usd"])
    total_jpy = float(emb_cost["jpy"]) + float(chat_cost["jpy"])

    # --------------------------------------------------------
    # 9. View を返す
    # --------------------------------------------------------
    return BotAnswerView(
        answer_text=answer,
        raw_hits=raw_hits,
        embedding_tokens=embedding_tokens,
        chat_input_tokens=chat_input_tokens,
        chat_output_tokens=chat_output_tokens,
        cost_usd=total_usd,
        cost_jpy=total_jpy,
        chat_model=chat_model,
        used_gemini=use_gemini,
    )
