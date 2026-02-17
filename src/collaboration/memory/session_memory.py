"""
会话记忆 - 多轮对话滑动窗口与 SQLite 持久化
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """单轮对话"""

    role: Literal["user", "assistant"]
    content: str
    intent: Optional[str] = None
    evidence_pack_id: Optional[str] = None
    canvas_patch: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


def _default_db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "sessions.db"


def _ensure_stage_column(conn: sqlite3.Connection) -> None:
    """兼容旧库：若 sessions 表无 stage 列则添加。"""
    cur = conn.execute("PRAGMA table_info(sessions)")
    columns = [row[1] for row in cur.fetchall()]
    if "stage" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN stage TEXT NOT NULL DEFAULT 'explore'")


def _ensure_citations_column(conn: sqlite3.Connection) -> None:
    """兼容旧库：若 turns 表无 citations_json 列则添加。"""
    cur = conn.execute("PRAGMA table_info(turns)")
    columns = [row[1] for row in cur.fetchall()]
    if "citations_json" not in columns:
        conn.execute("ALTER TABLE turns ADD COLUMN citations_json TEXT")


def _ensure_summary_columns(conn: sqlite3.Connection) -> None:
    """兼容旧库：若 sessions 表无滚动总结字段则添加。"""
    cur = conn.execute("PRAGMA table_info(sessions)")
    columns = [row[1] for row in cur.fetchall()]
    if "rolling_summary" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN rolling_summary TEXT NOT NULL DEFAULT ''")
    if "summary_at_turn" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN summary_at_turn INTEGER NOT NULL DEFAULT 0")


class SessionStore:
    """SQLite 持久化：会话与轮次"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    canvas_id TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL DEFAULT 'explore',
                    rolling_summary TEXT NOT NULL DEFAULT '',
                    summary_at_turn INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            _ensure_stage_column(conn)
            _ensure_summary_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    intent TEXT,
                    evidence_pack_id TEXT,
                    canvas_patch TEXT,
                    citations_json TEXT,
                    timestamp TEXT NOT NULL,
                    PRIMARY KEY (session_id, turn_index)
                )
                """
            )
            _ensure_citations_column(conn)
            conn.commit()

    def create_session(self, canvas_id: str = "", stage: str = "explore") -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            _ensure_stage_column(conn)
            _ensure_summary_columns(conn)
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, canvas_id, stage, rolling_summary, summary_at_turn, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, canvas_id, stage, "", 0, now, now),
            )
            conn.commit()
        return session_id

    def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            _ensure_stage_column(conn)
            _ensure_summary_columns(conn)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT session_id, canvas_id, stage, rolling_summary, summary_at_turn, created_at, updated_at
                FROM sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        meta = dict(row)
        if meta.get("stage") is None:
            meta["stage"] = "explore"
        return meta

    def get_session_stage(self, session_id: str) -> str:
        meta = self.get_session_meta(session_id)
        if meta is None:
            return "explore"
        return meta.get("stage") or "explore"

    def update_session_stage(self, session_id: str, stage: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            _ensure_stage_column(conn)
            conn.execute(
                "UPDATE sessions SET stage = ?, updated_at = ? WHERE session_id = ?",
                (stage, datetime.now().isoformat(), session_id),
            )
            conn.commit()

    def update_session_meta(self, session_id: str, meta: Dict[str, Any]) -> None:
        """更新会话元数据，如 canvas_id"""
        updates = []
        values: List[Any] = []
        if isinstance(meta.get("canvas_id"), str):
            updates.append("canvas_id = ?")
            values.append(meta["canvas_id"])
        if isinstance(meta.get("rolling_summary"), str):
            updates.append("rolling_summary = ?")
            values.append(meta["rolling_summary"])
        summary_at_turn = meta.get("summary_at_turn")
        if isinstance(summary_at_turn, int):
            updates.append("summary_at_turn = ?")
            values.append(summary_at_turn)
        if not updates:
            return
        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(session_id)
        sql = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?"
        with sqlite3.connect(self.db_path) as conn:
            _ensure_summary_columns(conn)
            conn.execute(sql, tuple(values))
            conn.commit()

    def get_turns(
        self,
        session_id: str,
        limit: Optional[int] = None,
        order_desc: bool = False,
    ) -> List[ConversationTurn]:
        with sqlite3.connect(self.db_path) as conn:
            _ensure_citations_column(conn)
            conn.row_factory = sqlite3.Row
            order = "DESC" if order_desc else "ASC"
            sql = (
                "SELECT role, content, intent, evidence_pack_id, canvas_patch, citations_json, timestamp "
                "FROM turns WHERE session_id = ? ORDER BY turn_index "
                + order
            )
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            rows = conn.execute(sql, (session_id,)).fetchall()
        turns = []
        for r in rows:
            patch = None
            if r["canvas_patch"]:
                try:
                    patch = json.loads(r["canvas_patch"])
                except Exception:
                    pass
            citations = []
            if r["citations_json"]:
                try:
                    citations = json.loads(r["citations_json"])
                except Exception:
                    pass
            ts = datetime.fromisoformat(r["timestamp"]) if r["timestamp"] else datetime.now()
            turns.append(
                ConversationTurn(
                    role=r["role"],
                    content=r["content"] or "",
                    intent=r["intent"],
                    evidence_pack_id=r["evidence_pack_id"],
                    canvas_patch=patch,
                    citations=citations,
                    timestamp=ts,
                )
            )
        if order_desc:
            turns.reverse()
        return turns

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        evidence_pack_id: Optional[str] = None,
        canvas_patch: Optional[Dict[str, Any]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        meta = self.get_session_meta(session_id)
        if meta is None:
            raise ValueError(f"Session not found: {session_id}")
        with sqlite3.connect(self.db_path) as conn:
            _ensure_citations_column(conn)
            cur = conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns WHERE session_id = ?",
                (session_id,),
            )
            idx = cur.fetchone()[0]
            conn.execute(
                """INSERT INTO turns (session_id, turn_index, role, content, intent, evidence_pack_id, canvas_patch, citations_json, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    idx,
                    role,
                    content,
                    intent,
                    evidence_pack_id,
                    json.dumps(canvas_patch, ensure_ascii=False) if canvas_patch else None,
                    json.dumps(citations, ensure_ascii=False) if citations else None,
                    datetime.now().isoformat(),
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )
            conn.commit()

    def delete_session(self, session_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        return cur.rowcount > 0

    def list_all_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有会话，按更新时间倒序"""
        with sqlite3.connect(self.db_path) as conn:
            _ensure_stage_column(conn)
            _ensure_summary_columns(conn)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT session_id, canvas_id, stage, rolling_summary, summary_at_turn, created_at, updated_at
                FROM sessions ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        sessions = []
        for row in rows:
            meta = dict(row)
            if meta.get("stage") is None:
                meta["stage"] = "explore"
            # 获取第一轮对话作为标题
            turns = self.get_turns(meta["session_id"], limit=1)
            meta["title"] = turns[0].content[:50] + "..." if turns and turns[0].content else "未命名对话"
            meta["turn_count"] = self._count_turns(meta["session_id"])
            sessions.append(meta)
        return sessions

    def _count_turns(self, session_id: str) -> int:
        """统计会话的轮次数"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,))
            return cur.fetchone()[0]


_store: Optional[SessionStore] = None


def get_session_store(db_path: Optional[Path] = None) -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore(db_path=db_path)
    return _store


@dataclass
class SessionMemory:
    """会话级短期记忆（滑动窗口，持久化由 SessionStore 负责）"""

    session_id: str
    canvas_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: int = 20
    rolling_summary: str = ""
    summary_at_turn: int = 0

    def add_turn(self, role: str, content: str, **kwargs: Any) -> None:
        turn = ConversationTurn(role=role, content=content, **kwargs)
        self.turns.append(turn)
        store = get_session_store()
        store.append_turn(
            self.session_id,
            role=role,
            content=content,
            intent=kwargs.get("intent"),
            evidence_pack_id=kwargs.get("evidence_pack_id"),
            canvas_patch=kwargs.get("canvas_patch"),
            citations=kwargs.get("citations"),
        )
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def get_context_window(self, n: int = 10) -> List[ConversationTurn]:
        return self.turns[-n:]

    def to_messages(self) -> List[Dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def update_rolling_summary(self, llm_client: Any, interval: int = 4) -> None:
        """Update rolling summary every `interval` turns."""
        current_count = len(self.turns)
        if current_count - self.summary_at_turn < max(1, interval):
            return

        recent_turns = self.turns[self.summary_at_turn:current_count]
        turns_text = "\n".join(
            f"{'User' if t.role == 'user' else 'Assistant'}: {(t.content or '')[:200]}"
            for t in recent_turns
        )
        prompt = f"""Summarize the following conversation segment in 2-3 sentences.
Focus on the main topic, key concepts discussed, and the user's current direction.

Previous summary:
{self.rolling_summary or "(start of conversation)"}

New conversation segment:
{turns_text}

Output only the updated summary."""
        try:
            resp = llm_client.chat(
                messages=[
                    {"role": "system", "content": "Output a concise conversation summary."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
            )
            text = (resp.get("final_text") or "").strip()
            if text:
                self.rolling_summary = text
                self.summary_at_turn = current_count
                get_session_store().update_session_meta(
                    self.session_id,
                    {
                        "rolling_summary": self.rolling_summary,
                        "summary_at_turn": self.summary_at_turn,
                    },
                )
        except Exception as e:
            logger.debug("rolling summary update failed: %s", e)


def load_session_memory(session_id: str, max_turns: int = 20) -> Optional[SessionMemory]:
    """从持久化加载会话记忆"""
    store = get_session_store()
    meta = store.get_session_meta(session_id)
    if meta is None:
        return None
    turns = store.get_turns(session_id, limit=max_turns * 2, order_desc=True)
    return SessionMemory(
        session_id=meta["session_id"],
        canvas_id=meta["canvas_id"] or "",
        turns=turns[-max_turns:] if len(turns) > max_turns else turns,
        max_turns=max_turns,
        rolling_summary=meta.get("rolling_summary") or "",
        summary_at_turn=int(meta.get("summary_at_turn") or 0),
    )
