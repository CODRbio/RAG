"""
全局 Debug Logger — 结构化 JSONL 文件 + console 双输出。

控制方式（优先级从高到低）：
1. API 热切换: POST /debug/toggle {"enabled": true}
2. 环境变量:   RAG_DEBUG=1
3. 配置文件:   config/rag_config.json → {"debug": true}

日志阶段 (stages):
  query_route     — 查询路由决策（正则/LLM 分类）
  query_build     — 检索 Query 改写
  retrieval       — 检索执行详情（评分、去重、来源）
  prompt_assembly — System Prompt 组装
  agent_loop      — ReAct 循环每轮 LLM 输入输出 + 工具调用
  llm_direct      — 直接 LLM 调用（非 Agent 模式）
  citation_resolve— 引文后处理、tools_contributed 判定
  turn_summary    — 整轮请求汇总
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "debug"
_MAX_AGE_DAYS = 7
_MAX_TOTAL_MB = 200


class DebugLogger:
    """
    全局结构化 Debug 日志。

    - 写入 ``logs/debug/YYYY-MM-DD.jsonl``（JSON Lines，每行一条记录）
    - 同时输出到 stderr（方便 ``tail -f`` 或终端实时观察）
    - ``enabled`` 标志可通过 ``toggle()`` 热切换，无需重启
    """

    def __init__(
        self,
        log_dir: Path | str | None = None,
        enabled: bool = False,
    ):
        self._lock = threading.Lock()
        self.enabled = enabled
        self.log_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._console = logging.getLogger("rag.debug")
        self._console.setLevel(logging.DEBUG)
        self._console.propagate = False
        if not self._console.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter("[DEBUG %(asctime)s] %(message)s", datefmt="%H:%M:%S")
            )
            self._console.addHandler(handler)

    # ── 核心 API ──

    def log_stage(
        self,
        stage: str,
        session_id: str,
        data: Dict[str, Any],
        *,
        message: str = "",
    ) -> None:
        """
        记录一个 pipeline 阶段的 debug 信息。

        Args:
            stage: 阶段名称 (query_route / retrieval / agent_loop 等)
            session_id: 会话 ID（用于追溯）
            data: 该阶段的详细数据
            message: 可选的简短描述（仅用于 console 输出）
        """
        if not self.enabled:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "session_id": session_id,
            "stage": stage,
            "data": data,
        }
        self._write_jsonl(record)
        summary = message or self._auto_summary(stage, data)
        self._console.debug("[%s] session=%s | %s", stage, session_id[:12], summary)

    def toggle(self, enabled: bool) -> None:
        """热切换 debug 模式（API 调用）。"""
        self.enabled = enabled
        self._console.info("Debug mode %s via API", "ENABLED" if enabled else "DISABLED")

    # ── 便捷方法 ──

    def log_query_route(self, session_id: str, **kwargs: Any) -> None:
        self.log_stage("query_route", session_id, kwargs)

    def log_query_build(self, session_id: str, **kwargs: Any) -> None:
        self.log_stage("query_build", session_id, kwargs)

    def log_retrieval(self, session_id: str, **kwargs: Any) -> None:
        self.log_stage("retrieval", session_id, kwargs)

    def log_prompt_assembly(self, session_id: str, **kwargs: Any) -> None:
        self.log_stage("prompt_assembly", session_id, kwargs)

    def log_agent_iteration(self, session_id: str, **kwargs: Any) -> None:
        self.log_stage("agent_loop", session_id, kwargs)

    def log_llm_direct(self, session_id: str, **kwargs: Any) -> None:
        self.log_stage("llm_direct", session_id, kwargs)

    def log_citation_resolve(self, session_id: str, **kwargs: Any) -> None:
        self.log_stage("citation_resolve", session_id, kwargs)

    def log_turn_summary(self, session_id: str, **kwargs: Any) -> None:
        self.log_stage("turn_summary", session_id, kwargs)

    # ── 内部方法 ──

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        path = self.log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    @staticmethod
    def _auto_summary(stage: str, data: Dict[str, Any]) -> str:
        """根据 stage 自动生成简短的 console 摘要。"""
        if stage == "query_route":
            return f"decision={data.get('decision')} latency={data.get('latency_ms', '?')}ms"
        if stage == "query_build":
            orig = str(data.get("original_message", ""))[:40]
            rewritten = str(data.get("rewritten_query", ""))[:40]
            return f"'{orig}' → '{rewritten}' ({data.get('latency_ms', '?')}ms)"
        if stage == "retrieval":
            return (
                f"mode={data.get('mode')} chunks={data.get('total_chunks', '?')} "
                f"sources={data.get('sources_used', [])} ({data.get('latency_ms', '?')}ms)"
            )
        if stage == "prompt_assembly":
            return f"mode={data.get('prompt_mode')} sys_len={data.get('system_content_len', '?')}"
        if stage == "agent_loop":
            return (
                f"iter={data.get('iteration', '?')} tool={data.get('tool_name', '-')} "
                f"tool_ms={data.get('tool_latency_ms', '?')} llm_ms={data.get('llm_latency_ms', '?')}"
            )
        if stage == "llm_direct":
            return f"tokens={data.get('tokens', '?')} ({data.get('latency_ms', '?')}ms)"
        if stage == "citation_resolve":
            return (
                f"citations={data.get('cited_count', '?')} "
                f"tools_contributed={data.get('tools_contributed', '?')}"
            )
        if stage == "turn_summary":
            return f"total={data.get('total_ms', '?')}ms route={data.get('route')} gen={data.get('gen_mode')}"
        return json.dumps(data, ensure_ascii=False, default=str)[:120]

    def cleanup(
        self,
        max_age_days: int = _MAX_AGE_DAYS,
        max_total_mb: int = _MAX_TOTAL_MB,
    ) -> Dict[str, Any]:
        """清理过期/过大的 debug 日志文件。"""
        from datetime import timedelta

        report: Dict[str, Any] = {"deleted_by_age": [], "deleted_by_size": [], "remaining_mb": 0.0}
        if not self.log_dir.exists():
            return report
        files = sorted(self.log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        cutoff = datetime.now() - timedelta(days=max_age_days)
        remaining = []
        for f in files:
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                report["deleted_by_age"].append(f.name)
                f.unlink()
            else:
                remaining.append(f)
        max_bytes = max_total_mb * 1024 * 1024
        while remaining:
            total = sum(f.stat().st_size for f in remaining)
            if total <= max_bytes:
                break
            oldest = remaining.pop(0)
            report["deleted_by_size"].append(oldest.name)
            oldest.unlink()
        if remaining:
            report["remaining_mb"] = sum(f.stat().st_size for f in remaining) / (1024 * 1024)
        return report


# ── 全局单例 ──

_instance: Optional[DebugLogger] = None
_init_lock = threading.Lock()


def init_debug_logger(
    log_dir: Path | str | None = None,
    enabled: bool | None = None,
) -> DebugLogger:
    """
    初始化全局 DebugLogger 单例。

    enabled 判定优先级: 参数 > 环境变量 RAG_DEBUG > 配置文件 debug 字段 > False
    """
    global _instance
    with _init_lock:
        if enabled is None:
            env_val = os.getenv("RAG_DEBUG", "").lower()
            if env_val in ("1", "true"):
                enabled = True
            else:
                try:
                    from config.settings import settings
                    enabled = getattr(settings, "debug", False)
                except Exception:
                    enabled = False
        _instance = DebugLogger(log_dir=log_dir, enabled=enabled)
        return _instance


def get_debug_logger() -> DebugLogger:
    """获取全局 DebugLogger 单例（未初始化时自动按默认配置创建）。"""
    global _instance
    if _instance is None:
        return init_debug_logger()
    return _instance
