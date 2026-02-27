"""
会话记忆 - 多轮对话滑动窗口与持久化。
底层存储已迁移至 data/rag.db (sessions / turns 表)，通过 SQLModel 访问。
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import ChatSession, Turn as TurnRow
from src.log import get_logger
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()
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


class SessionStore:
    """SQLModel 持久化：会话与轮次"""

    def __init__(self, db_path=None):
        # db_path ignored — we use the shared engine
        pass

    def create_session(self, canvas_id: str = "", stage: str = "explore") -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with Session(get_engine()) as session:
            row = ChatSession(
                session_id=session_id,
                canvas_id=canvas_id,
                stage=stage,
                rolling_summary="",
                summary_at_turn=0,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.commit()
        return session_id

    def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        with Session(get_engine()) as session:
            row = session.get(ChatSession, session_id)
        if row is None:
            return None
        meta = {
            "session_id": row.session_id,
            "canvas_id": row.canvas_id or "",
            "stage": row.stage or "explore",
            "rolling_summary": row.rolling_summary or "",
            "summary_at_turn": row.summary_at_turn or 0,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
        return meta

    def get_session_stage(self, session_id: str) -> str:
        meta = self.get_session_meta(session_id)
        if meta is None:
            return "explore"
        return meta.get("stage") or "explore"

    def update_session_stage(self, session_id: str, stage: str) -> None:
        with Session(get_engine()) as session:
            row = session.get(ChatSession, session_id)
            if row:
                row.stage = stage
                row.updated_at = datetime.now().isoformat()
                session.add(row)
                session.commit()

    def update_session_meta(self, session_id: str, meta: Dict[str, Any]) -> None:
        """更新会话元数据，如 canvas_id"""
        with Session(get_engine()) as session:
            row = session.get(ChatSession, session_id)
            if not row:
                return
            if isinstance(meta.get("canvas_id"), str):
                row.canvas_id = meta["canvas_id"]
            if isinstance(meta.get("rolling_summary"), str):
                row.rolling_summary = meta["rolling_summary"]
            if isinstance(meta.get("summary_at_turn"), int):
                row.summary_at_turn = meta["summary_at_turn"]
            row.updated_at = datetime.now().isoformat()
            session.add(row)
            session.commit()

    def get_turns(
        self,
        session_id: str,
        limit: Optional[int] = None,
        order_desc: bool = False,
    ) -> List[ConversationTurn]:
        with Session(get_engine()) as session:
            stmt = (
                select(TurnRow)
                .where(TurnRow.session_id == session_id)
                .order_by(TurnRow.turn_index.desc() if order_desc else TurnRow.turn_index.asc())
            )
            if limit is not None:
                stmt = stmt.limit(int(limit))
            rows = session.exec(stmt).all()

        turns = []
        for r in rows:
            patch = None
            if r.canvas_patch:
                try:
                    patch = json.loads(r.canvas_patch)
                except Exception:
                    pass
            citations = []
            if r.citations_json:
                try:
                    citations = json.loads(r.citations_json)
                except Exception:
                    pass
            ts = datetime.fromisoformat(r.timestamp) if r.timestamp else datetime.now()
            turns.append(
                ConversationTurn(
                    role=r.role,
                    content=r.content or "",
                    intent=r.intent,
                    evidence_pack_id=r.evidence_pack_id,
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
        with Session(get_engine()) as session:
            # Get next turn index atomically
            existing = session.exec(
                select(TurnRow)
                .where(TurnRow.session_id == session_id)
                .order_by(TurnRow.turn_index.desc())
                .limit(1)
            ).first()
            idx = (existing.turn_index + 1) if existing else 0

            turn = TurnRow(
                session_id=session_id,
                turn_index=idx,
                role=role,
                content=content,
                intent=intent,
                evidence_pack_id=evidence_pack_id,
                canvas_patch=json.dumps(canvas_patch, ensure_ascii=False) if canvas_patch else None,
                citations_json=json.dumps(citations, ensure_ascii=False) if citations else None,
                timestamp=datetime.now().isoformat(),
            )
            session.add(turn)

            # Update session's updated_at
            chat_session = session.get(ChatSession, session_id)
            if chat_session:
                chat_session.updated_at = datetime.now().isoformat()
                session.add(chat_session)

            session.commit()

    def delete_session(self, session_id: str) -> bool:
        with Session(get_engine()) as session:
            row = session.get(ChatSession, session_id)
            if not row:
                return False
            session.delete(row)  # cascade deletes turns
            session.commit()
        return True

    def list_all_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有会话，按更新时间倒序"""
        with Session(get_engine()) as session:
            stmt = (
                select(ChatSession)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
            )
            rows = session.exec(stmt).all()

        sessions = []
        for row in rows:
            meta = {
                "session_id": row.session_id,
                "canvas_id": row.canvas_id or "",
                "stage": row.stage or "explore",
                "rolling_summary": row.rolling_summary or "",
                "summary_at_turn": row.summary_at_turn or 0,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            turns = self.get_turns(meta["session_id"], limit=1)
            meta["title"] = turns[0].content[:50] + "..." if turns and turns[0].content else "未命名对话"
            meta["turn_count"] = self._count_turns(meta["session_id"])
            sessions.append(meta)
        return sessions

    def _count_turns(self, session_id: str) -> int:
        """统计会话的轮次数"""
        with Session(get_engine()) as session:
            from sqlmodel import func
            result = session.exec(
                select(func.count()).select_from(TurnRow).where(TurnRow.session_id == session_id)
            ).one()
            return result


_store: Optional[SessionStore] = None


def get_session_store(db_path=None) -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
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
            self.turns = self.turns[-self.max_turns:]

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
        prompt = _pm.render(
            "session_memory_summarize.txt",
            previous_summary=self.rolling_summary or "(start of conversation)",
            turns_text=turns_text,
        )
        try:
            resp = llm_client.chat(
                messages=[
                    {"role": "system", "content": _pm.render("session_memory_summarize_system.txt")},
                    {"role": "user", "content": prompt},
                ],

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
