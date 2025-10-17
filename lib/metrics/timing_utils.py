# lib/metrics/timing_utils.py
from __future__ import annotations
import time
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict
import streamlit as st

@dataclass
class Timings:
    """処理の節目（絶対時刻）と高分解能タイマーを記録し、差分も出せる小ユーティリティ。"""
    marks: Dict[str, dt.datetime] = field(default_factory=dict)  # wall clock
    perf: Dict[str, float] = field(default_factory=dict)        # perf_counter 秒

    def mark(self, name: str) -> None:
        self.marks[name] = dt.datetime.now()
        self.perf[name] = time.perf_counter()

    def elapsed(self, start: str, end: str) -> float:
        return max(0.0, self.perf.get(end, 0.0) - self.perf.get(start, 0.0))

    def when(self, name: str) -> str:
        t = self.marks.get(name)
        return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if t else "-"

def stream_with_timing(gen, timings: Timings, first_key: str = "gpt_first_token", done_key: str = "gpt_done"):
    """Streamlitのst.write_streamに渡すgeneratorを包み、最初のトークン到達／完了時刻を記録する。"""
    first_seen = False
    for chunk in gen:
        if not first_seen:
            timings.mark(first_key)
            first_seen = True
        yield chunk
    timings.mark(done_key)


def render_metrics_ui(timings) -> None:
    """タイミング可視化の描画（開始時刻と区間別経過秒）。"""
    st.markdown("### ⏱ 実行タイムライン（ms 精度）")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**開始時刻**")
        st.code(f"""
pipeline_start : {timings.when('pipeline_start')}
scan_start     : {timings.when('scan_start')}
embed_start    : {timings.when('embed_start')}
search_start   : {timings.when('search_start')}
gpt_req_start  : {timings.when('gpt_req_start') if 'gpt_req_start' in timings.marks else '-'}
gpt_first_token: {timings.when('gpt_first_token') if 'gpt_first_token' in timings.marks else '-'}
gpt_done       : {timings.when('gpt_done') if 'gpt_done' in timings.marks else '-'}
pipeline_end   : {timings.when('pipeline_end')}
""".strip())
    with col2:
        st.write("**区間別の経過秒**")
        st.code(f"""
candidate_scan : {timings.elapsed('scan_start', 'scan_end'):.3f} s
embedding      : {timings.elapsed('embed_start', 'embed_end'):.3f} s
vector_search  : {timings.elapsed('search_start', 'search_end'):.3f} s
gpt_wait       : {timings.elapsed('gpt_req_start', 'gpt_first_token'):.3f} s
gpt_stream     : {timings.elapsed('gpt_first_token', 'gpt_done'):.3f} s
gpt_total      : {timings.elapsed('gpt_req_start', 'gpt_done'):.3f} s
pipeline_total : {timings.elapsed('pipeline_start', 'pipeline_end'):.3f} s
""".strip())
    with col3:
        st.write("**メモ**")
        st.caption(
            "- *gpt_wait*: API 送信→最初のトークン到着まで\n"
            "- *gpt_stream*: 最初のトークン→完了まで\n"
            "- 一括表示では *gpt_wait* はほぼ *gpt_total* と同じ、"
            "*gpt_stream* は 0 に近くなります。"
        )
