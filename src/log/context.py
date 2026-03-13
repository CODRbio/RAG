"""
请求上下文：correlation_id 通过 ContextVar 在异步调用链中自动传播。

用法：
  # 在 middleware 或后台任务入口设置（记得用 try...finally 重置）
  from src.log.context import set_correlation_id, reset_correlation_id
  token = set_correlation_id("req-abc123")
  try:
      # ...你的业务逻辑...
      pass
  finally:
      reset_correlation_id(token)

  # 在任意模块读取（无需传参）
  from src.log.context import get_correlation_id
  cid = get_correlation_id()   # → "req-abc123"
"""
from contextvars import ContextVar, Token
import uuid

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


def get_correlation_id() -> str:
    return _correlation_id.get()


def set_correlation_id(cid: str) -> Token[str]:
    """设置 correlation_id，返回 Token 供重置使用（防止线程池/后台任务泄漏）。"""
    return _correlation_id.set(cid)


def reset_correlation_id(token: Token[str]) -> None:
    """恢复上一级的 correlation_id。"""
    _correlation_id.reset(token)


def new_correlation_id() -> str:
    """生成并设置一个新的 correlation_id，返回该值。"""
    cid = f"req-{uuid.uuid4().hex[:8]}"
    _correlation_id.set(cid)
    return cid
