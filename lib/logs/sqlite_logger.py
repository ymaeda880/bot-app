# lib/logs/sqlite_logger.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import sqlite3
from datetime import datetime, timezone, timedelta

# この名前は 40_ログ管理.py 側の想定とも揃えておく
DB_FILE_NAME = "bot_logs.sqlite3"


def get_db_path(app_dir: Path) -> Path:
    """
    app_dir（例：.../Storages/<sub>/bot_app）を受け取り、
    その配下の logs/bot_logs.sqlite3 を返す。
    """
    return app_dir / "logs" / DB_FILE_NAME


def init_bot_logs_db(app_dir: Path) -> Path:
    """
    bot_logs.sqlite3 を初期化（なければ作成）して Path を返す。

    重要（方針）:
      - この関数はディレクトリ作成（mkdir）をしない。
      - 呼び出し側（ページ側）が事前に logs/ を作成しておくこと。
    """
    db_path = get_db_path(app_dir)

    if not db_path.parent.exists():
        raise FileNotFoundError(
            f"logs ディレクトリが存在しません：{db_path.parent}（ページ側で mkdir してください）"
        )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 複数同時アクセスにそこそこ強くする設定
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,         -- JST ISO8601
            app TEXT NOT NULL,        -- 例: "bot_app"
            page TEXT NOT NULL,       -- 例: "13_ボット（ログ管理拡張版）__openai"
            user TEXT NOT NULL,       -- ログインユーザー
            action TEXT NOT NULL,     -- "ask" / "answer" など
            model TEXT,               -- 例: "gpt-5-mini"
            detail TEXT,              -- concise / standard / detailed...
            embedding_tokens INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            cost_jpy REAL,
            prompt_hash TEXT,
            answer_hash TEXT,
            prompt TEXT,
            answer TEXT
        );
        """
    )

    conn.commit()
    conn.close()
    return db_path


def insert_bot_log_row(
    db_path: Path,
    *,
    app: str,
    page: str,
    user: str,
    action: str,
    model: Optional[str] = None,
    detail: Optional[str] = None,
    embedding_tokens: Optional[int] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    cost_jpy: Optional[float] = None,
    prompt_hash: Optional[str] = None,
    answer_hash: Optional[str] = None,
    prompt: Optional[str] = None,
    answer: Optional[str] = None,
) -> None:
    """
    logs テーブルに 1 行挿入する共通関数。
    """
    JST = timezone(timedelta(hours=9))
    conn = sqlite3.connect(db_path, timeout=5.0)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO logs (
            ts, app, page, user, action, model, detail,
            embedding_tokens, input_tokens, output_tokens,
            cost_usd, cost_jpy,
            prompt_hash, answer_hash, prompt, answer
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(JST).isoformat(),
            app,
            page,
            user,
            action,
            model,
            detail,
            embedding_tokens,
            input_tokens,
            output_tokens,
            cost_usd,
            cost_jpy,
            prompt_hash,
            answer_hash,
            prompt,
            answer,
        ),
    )

    conn.commit()
    conn.close()


def preview_text(text: str, length: int = 20) -> str:
    """テキストの先頭 length 文字を返すだけの小ヘルパー。"""
    if not text:
        return ""
    return text[:length]
