"""
会话记忆 - 多轮对话滑动窗口与持久化。
底层存储已迁移至 data/rag.db (sessions / turns 表)，通过 SQLModel 访问。
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import ChatSession, Turn as TurnRow
from src.log import get_logger
from src.utils.prompt_manager import PromptManager
from src.utils.context_limits import SESSION_MEMORY_TURN_MAX_CHARS

_pm = PromptManager()
logger = get_logger(__name__)

# --- 记忆系统核心参数配置 (Memory System Constants) ---
# 短期记忆缓冲区上限 (约 10k tokens)
MAX_BUFFER_CHARS = 40000 
# 滚动摘要硬上限 (约 3k tokens)
MAX_SUMMARY_CHARS = 12000
# 置顶文献块容量上限 (约 7.5k tokens)
MAX_PINNED_CHARS = 30000
# 证据缓存保留的检索轮次
MAX_EVIDENCE_TURNS = 10
# 证据块单体最大字符数
_EVIDENCE_CACHE_MAX_TEXT_CHARS = 3000


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

    def create_session(
        self,
        canvas_id: str = "",
        stage: str = "explore",
        session_type: str = "chat",
        user_id: str = "",
    ) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        def _do() -> None:
            with Session(get_engine()) as session:
                row = ChatSession(
                    session_id=session_id,
                    user_id=user_id or "",
                    canvas_id=canvas_id,
                    stage=stage,
                    rolling_summary="",
                    summary_at_turn=0,
                    title="",
                    session_type=session_type if session_type in ("chat", "research") else "chat",
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                session.commit()

        _do()
        return session_id

    def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        with Session(get_engine()) as session:
            row = session.get(ChatSession, session_id)
        if row is None:
            return None
        preferences = {}
        raw_prefs = getattr(row, "preferences", None) or ""
        if raw_prefs:
            try:
                preferences = json.loads(raw_prefs)
            except (TypeError, ValueError):
                pass
        meta = {
            "session_id": row.session_id,
            "user_id": getattr(row, "user_id", "") or "",
            "canvas_id": row.canvas_id or "",
            "stage": row.stage or "explore",
            "rolling_summary": row.rolling_summary or "",
            "summary_at_turn": row.summary_at_turn or 0,
            "title": getattr(row, "title", None) or "",
            "session_type": getattr(row, "session_type", None) or "chat",
            "preferences": preferences,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
        return meta

    def get_session_owner(self, session_id: str) -> Optional[str]:
        meta = self.get_session_meta(session_id)
        if meta is None:
            return None
        return str(meta.get("user_id") or "").strip() or None

    def get_session_stage(self, session_id: str) -> str:
        meta = self.get_session_meta(session_id)
        if meta is None:
            return "explore"
        return meta.get("stage") or "explore"

    def update_session_stage(self, session_id: str, stage: str) -> None:
        def _do() -> None:
            with Session(get_engine()) as session:
                row = session.get(ChatSession, session_id)
                if row:
                    row.stage = stage
                    row.updated_at = datetime.now().isoformat()
                    session.add(row)
                    session.commit()

        _do()

    def update_session_meta(self, session_id: str, meta: Dict[str, Any]) -> None:
        """更新会话元数据，如 canvas_id、title、session_type"""

        def _do() -> None:
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
                if "title" in meta and isinstance(meta.get("title"), str):
                    row.title = meta["title"]
                if meta.get("session_type") in ("chat", "research"):
                    row.session_type = meta["session_type"]
                if "preferences" in meta and isinstance(meta["preferences"], dict):
                    existing = {}
                    if getattr(row, "preferences", None):
                        try:
                            existing = json.loads(row.preferences)
                        except (TypeError, ValueError):
                            pass
                    existing.update(meta["preferences"])
                    row.preferences = json.dumps(existing, ensure_ascii=False)
                row.updated_at = datetime.now().isoformat()
                session.add(row)
                session.commit()

        _do()

    def get_recent_evidence_cache(self, session_id: str) -> List[Dict[str, Any]]:
        """读取会话内最近证据缓存（含命中提权与置顶逻辑）。"""
        meta = self.get_session_meta(session_id)
        if not meta:
            return []
        prefs = meta.get("preferences") or {}
        raw = prefs.get("recent_evidence_cache")
        if not isinstance(raw, list):
            return []
            
        # 此时返回的应是过滤后的活跃项
        return [item for item in raw if isinstance(item, dict) and item.get("chunks")]

    def pin_evidence_chunks(self, session_id: str, chunks: List[Dict[str, Any]]) -> None:
        """将特定文献块标记为“置顶(Pinned)”，不受常规滑动窗口淘汰影响。"""
        if not session_id or not chunks:
            return
        
        meta = self.get_session_meta(session_id)
        if not meta: return
        prefs = meta.get("preferences") or {}
        cache = prefs.get("recent_evidence_cache") or []
        
        pinned_entry = {
            "query": "[PINNED_CONTEXT]",
            "timestamp": datetime.now().isoformat(),
            "is_pinned": True,
            "chunks": [self._sanitize_cached_chunk(c) for c in chunks]
        }
        
        # Pinned Context 容量熔断：检查总置顶字符数
        current_pinned_chars = sum(len(str(c.get("text", ""))) for item in cache if item.get("is_pinned") for c in item.get("chunks", []))
        new_chars = sum(len(str(c.get("text", ""))) for c in pinned_entry["chunks"])
        
        if current_pinned_chars + new_chars > MAX_PINNED_CHARS:
            logger.warning("[SessionStore] Pinned context limit reached (%d chars), rejecting new pins for session %s", MAX_PINNED_CHARS, session_id)
            return

        cache.insert(0, pinned_entry)
        self.update_session_meta(session_id, {"preferences": {"recent_evidence_cache": cache}})

    def boost_evidence_hit(self, session_id: str, chunk_ids: List[str]) -> None:
        """
        命中提权：当 AI 实际引用了某些 Chunk 时，增加其命中计数，延长其在缓存中的生命周期。
        """
        if not session_id or not chunk_ids:
            return
            
        meta = self.get_session_meta(session_id)
        if not meta: return
        prefs = meta.get("preferences") or {}
        cache = prefs.get("recent_evidence_cache") or []
        
        updated = False
        target_ids = set(chunk_ids)
        for item in cache:
            for c in item.get("chunks", []):
                if c.get("chunk_id") in target_ids:
                    c["hit_count"] = c.get("hit_count", 0) + 1
                    # 每次命中，相当于赋予该条目一次“续命”机会（即便它很旧）
                    item["timestamp"] = datetime.now().isoformat()
                    updated = True
        
        if updated:
            self.update_session_meta(session_id, {"preferences": {"recent_evidence_cache": cache}})

    def append_recent_evidence_cache(
        self,
        session_id: str,
        query: str,
        chunks: List[Dict[str, Any]],
        *,
        max_turns: int = MAX_EVIDENCE_TURNS,
        max_chunks_per_turn: int = 36,
    ) -> None:
        """追加证据缓存，支持按命中权重(Hit Boosting)和置顶(Pinned)进行智能化淘汰。"""
        if not session_id or not chunks:
            return
        
        meta = self.get_session_meta(session_id)
        if not meta: return
        prefs = meta.get("preferences") or {}
        existing = prefs.get("recent_evidence_cache") or []
        
        cleaned_chunks = [
            self._sanitize_cached_chunk(c)
            for c in chunks[: max(1, int(max_chunks_per_turn))]
            if isinstance(c, dict)
        ]
        cleaned_chunks = [c for c in cleaned_chunks if c]
        if not cleaned_chunks:
            return

        new_entry = {
            "query": (query or "").strip()[:500],
            "timestamp": datetime.now().isoformat(),
            "chunks": cleaned_chunks,
            "is_pinned": False
        }
        existing.append(new_entry)
        
        # 智能化淘汰算法：
        # 1. 置顶项 (is_pinned=True) 永远保留，除非手动移除。
        # 2. 普通项按 (max_turns) 限制淘汰，但高命中项 (hit_count > 0) 获得豁免权，多留一轮。
        
        pinned_items = [item for item in existing if item.get("is_pinned")]
        normal_items = [item for item in existing if not item.get("is_pinned")]
        
        def _get_item_weight(item):
            # 基础权重是时间，命中次数作为增益
            max_hit = max([c.get("hit_count", 0) for c in item.get("chunks", [])] or [0])
            return max_hit

        # 仅对普通项进行截断
        if len(normal_items) > max_turns:
            # 排序：优先保留命中次数高的，其次保留最近的
            normal_items.sort(key=lambda x: (_get_item_weight(x), x.get("timestamp")), reverse=True)
            normal_items = normal_items[:max_turns]
            
        final_cache = pinned_items + normal_items
        self.update_session_meta(
            session_id,
            {"preferences": {"recent_evidence_cache": final_cache}},
        )

    @staticmethod
    def _sanitize_cached_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce cached chunk payload size while keeping rerank-relevant fields."""
        text = str(chunk.get("text") or "")
        if len(text) > _EVIDENCE_CACHE_MAX_TEXT_CHARS:
            text = text[:_EVIDENCE_CACHE_MAX_TEXT_CHARS]
        return {
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "doc_id": str(chunk.get("doc_id") or ""),
            "text": text,
            "score": float(chunk.get("score") or 0.0),
            "hit_count": int(chunk.get("hit_count", 0)),
            "source_type": str(chunk.get("source_type") or "dense"),
            "doc_title": str(chunk.get("doc_title") or "") or None,
            "authors": list(chunk.get("authors") or []) if isinstance(chunk.get("authors"), list) else None,
            "year": int(chunk["year"]) if isinstance(chunk.get("year"), int) else None,
            "url": str(chunk.get("url") or "") or None,
            "doi": str(chunk.get("doi") or "") or None,
            "page_num": int(chunk["page_num"]) if isinstance(chunk.get("page_num"), int) else None,
            "section_title": str(chunk.get("section_title") or "") or None,
            "evidence_type": str(chunk.get("evidence_type") or "") or None,
            "bbox": list(chunk.get("bbox") or []) if isinstance(chunk.get("bbox"), list) else None,
            "provider": str(chunk.get("provider") or "") or None,
        }

    def touch_session(self, session_id: str) -> None:
        """仅更新 updated_at，用于「重新激活」后使会话排到历史列表最前。"""

        def _do() -> None:
            with Session(get_engine()) as session:
                row = session.get(ChatSession, session_id)
                if row:
                    row.updated_at = datetime.now().isoformat()
                    session.add(row)
                    session.commit()

        _do()

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
            raw_ts = getattr(r, "timestamp", None)
            try:
                ts = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now()
            except (TypeError, ValueError):
                ts = datetime.now()
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
        max_retries = 3
        for attempt in range(max_retries):
            with Session(get_engine()) as session:
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

                chat_session = session.get(ChatSession, session_id)
                if chat_session:
                    chat_session.updated_at = datetime.now().isoformat()
                    session.add(chat_session)

                try:
                    session.commit()
                    return
                except IntegrityError:
                    session.rollback()
                    if attempt == max_retries - 1:
                        raise
                    logger.debug("[session_memory] append_turn index conflict, retry %s/%s", attempt + 1, max_retries)

    def delete_session(self, session_id: str) -> bool:
        def _do() -> bool:
            with Session(get_engine()) as session:
                row = session.get(ChatSession, session_id)
                if not row:
                    return False
                session.delete(row)  # cascade deletes turns
                session.commit()
            return True

        return _do()

    def delete_sessions_by_canvas_id(self, canvas_id: str) -> int:
        """删除该画布下的所有会话（及其轮次），返回删除的会话数。"""
        if not canvas_id:
            return 0
        with Session(get_engine()) as session:
            stmt = select(ChatSession).where(ChatSession.canvas_id == canvas_id)
            rows = list(session.exec(stmt).all())
            for row in rows:
                session.delete(row)  # cascade deletes turns
            session.commit()
        return len(rows)

    def list_all_sessions(self, limit: int = 100, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出会话，按更新时间倒序；可选按 owner 过滤。"""
        with Session(get_engine()) as session:
            stmt = select(ChatSession)
            if user_id is not None:
                stmt = stmt.where(ChatSession.user_id == user_id)
            stmt = stmt.order_by(ChatSession.updated_at.desc()).limit(limit)
            rows = session.exec(stmt).all()

        sessions = []
        for row in rows:
            meta = {
                "session_id": row.session_id,
                "user_id": getattr(row, "user_id", "") or "",
                "canvas_id": row.canvas_id or "",
                "stage": row.stage or "explore",
                "rolling_summary": row.rolling_summary or "",
                "summary_at_turn": row.summary_at_turn or 0,
                "title": getattr(row, "title", None) or "",
                "session_type": getattr(row, "session_type", None) or "chat",
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            if not meta["title"]:
                turns = self.get_turns(meta["session_id"], limit=1)
                meta["title"] = (
                    (turns[0].content[:50] + "..." if len((turns[0].content or "")) > 50 else (turns[0].content or "").strip())
                    if turns and turns[0].content
                    else "未命名对话"
                )
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

    def get_turn_count(self, session_id: str) -> int:
        """返回会话当前轮次数（公开接口）"""
        return self._count_turns(session_id)


_store: Optional[SessionStore] = None


def get_session_store(db_path=None) -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


@dataclass
class SessionMemory:
    """会话级短期记忆（基于 Token/Char 驱动的滑动驱逐与摘要，零重叠上下文）"""

    session_id: str
    canvas_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: int = 100  # 适当放大，主要依赖 max_buffer_chars 进行驱逐
    rolling_summary: str = ""
    summary_at_turn: int = 0

    def add_turn(self, role: str, content: str, **kwargs: Any) -> None:
        # Assistant 回答“瘦身”：剥离大块的思考过程或原始引用块
        import re
        if role == "assistant" and content:
            # 剔除可能包含的 <think> 标签内容
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            # 剔除可能包含的 <search_results> 标签内容
            content = re.sub(r"<search_results>.*?</search_results>", "", content, flags=re.DOTALL)
            # 剔除可能包含的 <evidence> 标签内容
            content = re.sub(r"<evidence>.*?</evidence>", "", content, flags=re.DOTALL)
            content = content.strip()

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
        
        # 兜底：防止 turns 列表无限增长导致的内存泄露
        if len(self.turns) > self.max_turns:
            over = len(self.turns) - self.max_turns
            self.turns = self.turns[over:]
            self.summary_at_turn = max(0, self.summary_at_turn - over)

    def get_context_window(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """返回尚未被归纳的活跃窗口对话（零重叠）。"""
        active_turns = self.turns[self.summary_at_turn:]
        if n is not None and n > 0:
            return active_turns[-n:]
        return active_turns

    def to_messages(self) -> List[Dict[str, str]]:
        """将活跃窗口内的对话转换为 LLM 消息格式。"""
        return [{"role": t.role, "content": t.content} for t in self.turns[self.summary_at_turn:]]

    def update_rolling_summary(
        self, llm_client: Any, interval: int = 4, ultra_lite: bool = True,
    ) -> None:
        """Token-based Summary Buffer 驱逐与摘要逻辑。"""
        # 注意：这里的 interval 参数保留是为了兼容现有 API 签名，但实际触发逻辑已改为容量驱动
        active_turns = self.turns[self.summary_at_turn:]
        
        def _calc_chars(turns: List[ConversationTurn]) -> int:
            return sum(len(t.content or "") for t in turns)
            
        if _calc_chars(active_turns) <= MAX_BUFFER_CHARS:
            return

        # 严格时序滑动驱逐：从 active_turns 头部移出轮次，直到剩余小于等于 MAX_BUFFER_CHARS
        evicted_turns = []
        while len(active_turns) > 1 and _calc_chars(active_turns) > MAX_BUFFER_CHARS:
            evicted_turns.append(active_turns.pop(0))
            self.summary_at_turn += 1
            
        if not evicted_turns:
            # Edge case: 单轮极其巨大，强制不驱逐最新一轮，保留上下文连贯性
            logger.warning("[SessionMemory] Single turn exceeds max_buffer_chars (%d), skipping eviction to preserve context.", MAX_BUFFER_CHARS)
            return

        turns_text = "\n".join(
            f"{'User' if t.role == 'user' else 'Assistant'}: {(t.content or '')[:SESSION_MEMORY_TURN_MAX_CHARS]}"
            for t in evicted_turns
        )
        
        # 呼叫 LLM 进行摘要合并与再压缩 (Re-compression)
        if ultra_lite:
            prompt = _pm.render(
                "session_memory_summarize_ultra_lite.txt",
                previous_summary=self.rolling_summary or "(无早期历史)",
                turns_text=turns_text,
            )
            system_content = _pm.render("session_memory_summarize_ultra_lite_system.txt")
        else:
            prompt = _pm.render(
                "session_memory_summarize.txt",
                previous_summary=self.rolling_summary or "(无早期历史)",
                turns_text=turns_text,
            )
            system_content = _pm.render("session_memory_summarize_system.txt")
            
        try:
            resp = llm_client.chat(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
            )
            text = (resp.get("final_text") or "").strip()
            if text:
                # 对返回的摘要强制加上硬上限防膨胀
                if len(text) > MAX_SUMMARY_CHARS:
                    logger.warning("[SessionMemory] Rolling summary exceeded %d chars, truncating.", MAX_SUMMARY_CHARS)
                    text = text[:MAX_SUMMARY_CHARS] + "..."
                    
                self.rolling_summary = text
                get_session_store().update_session_meta(
                    self.session_id,
                    {
                        "rolling_summary": self.rolling_summary,
                        "summary_at_turn": self.summary_at_turn,
                    },
                )
        except Exception as e:
            logger.warning("rolling summary update failed: %s", e)
            # 失败时回退 summary_at_turn，以便下次触发
            self.summary_at_turn -= len(evicted_turns)


def load_session_memory(session_id: str, max_turns: int = 100) -> Optional[SessionMemory]:
    """从持久化加载会话记忆"""
    store = get_session_store()
    meta = store.get_session_meta(session_id)
    if meta is None:
        return None
    
    # 获取最新的 1000 轮（在 DB 中倒序获取，返回时内部已逆序为时间正序），确保长会话时取到的是最新记忆
    turns = store.get_turns(session_id, limit=1000, order_desc=True)
    summary_at_turn = int(meta.get("summary_at_turn") or 0)
    
    # 因为现在只取最新 1000 轮，如果总轮次 > 1000，绝对索引会失效。
    # 重新校准 summary_at_turn: 
    # 如果系统有 1500 轮，最新 1000 轮的绝对索引被截断了，需要按相对位置计算
    total_db_turns = meta.get("turn_count", store.get_turn_count(session_id))
    if total_db_turns > 1000:
        # DB总数超过1000，取出的 turns 实际上是全集的最后1000个。
        # 如果 summary_at_turn (基于全集) < (total - 1000)，说明未归纳的全在1000内或者早就被归纳了
        shifted_summary = max(0, summary_at_turn - (total_db_turns - 1000))
        summary_at_turn = shifted_summary

    if summary_at_turn > len(turns):
        summary_at_turn = len(turns)
        
    return SessionMemory(
        session_id=meta["session_id"],
        canvas_id=meta["canvas_id"] or "",
        turns=turns,
        max_turns=1000,  # 内存中不再严格按轮次剔除，而是通过 Token 驱逐
        rolling_summary=meta.get("rolling_summary") or "",
        summary_at_turn=summary_at_turn,
    )
