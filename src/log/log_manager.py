"""
日志管理模块 v2：分层路由 + correlation_id 注入 + error 聚合 + 自动清理。

Logger 命名约定（src.* 模块路径自动路由到对应日志目录）：
  src.api.*            → logs/api/
  src.llm.*            → logs/llm/
  src.collaboration.*  → logs/agent/
  src.retrieval.*      ┐
  src.indexing.*       ├─ logs/service/
  src.chunking.*       │
  src.parser.*         ┘
  其他 src.*           → logs/system/
  全部 ERROR+          → logs/error/（跨层聚合，便于告警）

每层按天生成一个日志文件（YYYY-MM-DD.log），进程重启不新建文件，方便长期追踪。

correlation_id 通过 ContextVar 自动注入到每条日志，无需手动传递。
环境变量 RAG_LOG_LEVEL 可覆盖配置文件级别（DEBUG/INFO/WARNING/ERROR）。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ── 默认配置 ──────────────────────────────────────────────────────────────────
DEFAULT_LEVEL = "INFO"
DEFAULT_MAX_SIZE_MB = 100
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_MIN_KEEP_MB = 20
DEFAULT_CONSOLE_OUTPUT = True

# logger 名称前缀 → 日志子目录映射（按顺序匹配，最长前缀优先）
_LAYER_ROUTING: list[tuple[str, str]] = [
    ("src.api",           "api"),
    ("src.llm",           "llm"),
    ("src.collaboration", "agent"),
    ("src.retrieval",     "service"),
    ("src.indexing",      "service"),
    ("src.chunking",      "service"),
    ("src.parser",        "service"),
]
_DEFAULT_LAYER = "system"

# 日志格式：timestamp | level | correlation_id | logger | message
_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(correlation_id)-20s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _resolve_layer(name: str) -> str:
    """根据 logger 名称解析所属分层目录。"""
    for prefix, layer in _LAYER_ROUTING:
        if name == prefix or name.startswith(prefix + "."):
            return layer
    return _DEFAULT_LAYER


from src.log.context import get_correlation_id

# ── ContextVar Filter ─────────────────────────────────────────────────────────

class _CorrelationFilter(logging.Filter):
    """将当前 ContextVar 中的 correlation_id 注入到每条 LogRecord。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.correlation_id = get_correlation_id()
        except Exception:
            record.correlation_id = "-"
        return True


class DailyFileHandler(logging.FileHandler):
    """每天自动轮转的 FileHandler，按 YYYY-MM-DD.log 命名。"""
    
    def __init__(self, base_dir: Path, encoding: str = "utf-8"):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        super().__init__(self._get_current_file(), encoding=encoding)

    def _get_current_file(self) -> Path:
        return self.base_dir / f"{self.current_date}.log"

    def emit(self, record: logging.LogRecord) -> None:
        new_date = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d")
        if new_date != self.current_date:
            self.acquire()
            try:
                # 再次检查，防止多线程同时进入
                new_date_locked = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d")
                if new_date_locked != self.current_date:
                    self.current_date = new_date_locked
                    self.close()
                    self.baseFilename = os.path.abspath(self._get_current_file())
                    self.stream = self._open()
            finally:
                self.release()
        super().emit(record)


# ── LogManager ────────────────────────────────────────────────────────────────

class LogManager:
    """
    统一日志管理：
    - 分层路由到独立日志文件（api / llm / agent / service / system）
    - 错误日志额外写入 logs/error/ 聚合文件
    - 每条日志自动注入 correlation_id
    - 控制台 + 文件双输出
    - 按配置自动清理（最大 100MB，30天，低于 20MB 不删）
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        base = Path(__file__).resolve().parent.parent.parent
        self._log_root = (
            Path(config["log_dir"]) if config.get("log_dir") else base / "logs"
        )

        self.max_size_mb = int(config.get("max_size_mb", DEFAULT_MAX_SIZE_MB))
        self.max_age_days = int(config.get("max_age_days", DEFAULT_MAX_AGE_DAYS))
        self.min_keep_mb = int(config.get("min_keep_mb", DEFAULT_MIN_KEEP_MB))
        self.console_output = config.get("console_output", DEFAULT_CONSOLE_OUTPUT)

        level_name = (
            os.environ.get("RAG_LOG_LEVEL") or config.get("level") or DEFAULT_LEVEL
        ).upper()
        self.level = getattr(logging, level_name, logging.INFO)

        self._formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)
        self._corr_filter = _CorrelationFilter()

        # 每层 + error 聚合的文件 handler 缓存（按 layer 名）
        self._file_handlers: dict[str, logging.Handler] = {}
        # 记录已管理的 logger，便于重新初始化时刷新配置
        self._managed_loggers: set[logging.Logger] = set()

    # ── 内部：获取/创建文件 handler ──

    def _get_file_handler(self, layer: str) -> logging.Handler:
        if layer not in self._file_handlers:
            d = self._log_root / layer
            fh = DailyFileHandler(d, encoding="utf-8")
            fh.setLevel(self.level)
            fh.setFormatter(self._formatter)
            fh.addFilter(self._corr_filter)
            self._file_handlers[layer] = fh
        return self._file_handlers[layer]

    def _get_error_handler(self) -> logging.Handler:
        """所有 ERROR+ 统一写入 logs/error/ 聚合文件。"""
        if "error" not in self._file_handlers:
            d = self._log_root / "error"
            fh = DailyFileHandler(d, encoding="utf-8")
            fh.setLevel(logging.ERROR)
            fh.setFormatter(self._formatter)
            fh.addFilter(self._corr_filter)
            self._file_handlers["error"] = fh
        return self._file_handlers["error"]

    # ── 公共：获取具名 logger ──

    def get_logger(self, name: str) -> logging.Logger:
        """获取具名 logger，自动路由到对应层日志文件并绑定 correlation_id 注入。"""
        logger = logging.getLogger(name)
        if logger in self._managed_loggers and logger.handlers:
            return logger

        self._configure_logger(logger, name)
        self._managed_loggers.add(logger)
        return logger

    def _configure_logger(self, logger: logging.Logger, name: str) -> None:
        """为 logger 清理并重新绑定 handler"""
        logger.handlers.clear()
        layer = _resolve_layer(name)
        logger.setLevel(self.level)
        logger.propagate = False

        # 控制台 handler
        if self.console_output:
            ch = logging.StreamHandler()
            ch.setLevel(self.level)
            ch.setFormatter(self._formatter)
            ch.addFilter(self._corr_filter)
            logger.addHandler(ch)

        # 分层文件 handler
        logger.addHandler(self._get_file_handler(layer))

        # ERROR 聚合 handler（非 error 层才需要额外添加，避免重复写入）
        if layer != "error":
            logger.addHandler(self._get_error_handler())

    # ── 清理 ──────────────────────────────────────────────────────────────────

    def cleanup(self) -> dict[str, Any]:
        """
        清理所有层日志文件。
        规则：总大小 < min_keep_mb 时不删；先删超龄，再按最旧优先删至 max_size_mb。
        """
        report: dict[str, Any] = {
            "deleted_by_age": [],
            "deleted_by_size": [],
            "remaining_mb": 0.0,
        }
        if not self._log_root.exists():
            return report

        all_files: list[Path] = []
        for sub in self._log_root.iterdir():
            if sub.is_dir():
                all_files.extend(f for f in sub.iterdir() if f.is_file() and f.suffix == ".log")

        all_files.sort(key=lambda p: p.stat().st_mtime)
        min_bytes = self.min_keep_mb * 1024 * 1024
        total = sum(f.stat().st_size for f in all_files)
        if total < min_bytes:
            report["remaining_mb"] = total / (1024 * 1024)
            return report

        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        remaining: list[Path] = []
        for f in all_files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                report["deleted_by_age"].append(f.name)
                f.unlink(missing_ok=True)
            else:
                remaining.append(f)

        max_bytes = self.max_size_mb * 1024 * 1024
        
        # Calculate total size once
        current_total = sum(f.stat().st_size for f in remaining)
        
        while remaining and current_total > max_bytes:
            oldest = remaining.pop(0)
            report["deleted_by_size"].append(oldest.name)
            try:
                size = oldest.stat().st_size
                oldest.unlink(missing_ok=True)
                current_total -= size
            except OSError:
                pass

        if remaining:
            report["remaining_mb"] = current_total / (1024 * 1024)
        return report


# ── 模块级单例 ────────────────────────────────────────────────────────────────

_manager: LogManager | None = None


def _get_manager(config: dict[str, Any] | None = None) -> LogManager:
    global _manager
    if _manager is None:
        _manager = LogManager(config)
    return _manager


def _load_logging_config(path: Path) -> dict[str, Any]:
    import json
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = raw.get("logging") or {}
    local_path = path.with_name(f"{path.stem}.local{path.suffix}")
    if local_path.exists():
        local_raw = json.loads(local_path.read_text(encoding="utf-8"))
        local_cfg = local_raw.get("logging") or {}
        if local_cfg:
            cfg = {**cfg, **local_cfg}
    return cfg


def init_logging(
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> LogManager:
    """初始化日志管理器（可选从配置文件加载）。"""
    cfg = config
    if cfg is None and config_path is not None:
        cfg = _load_logging_config(Path(config_path))
    elif cfg is None:
        default_path = (
            Path(__file__).resolve().parent.parent.parent / "config" / "rag_config.json"
        )
        cfg = _load_logging_config(default_path)
    global _manager
    old_manager = _manager
    _manager = LogManager(cfg)
    
    if old_manager:
        for logger in old_manager._managed_loggers:
            _manager._configure_logger(logger, logger.name)
            _manager._managed_loggers.add(logger)
            
    return _manager


def get_logger(name: str, config: dict[str, Any] | None = None) -> logging.Logger:
    """获取 logger。尚未初始化时自动从 rag_config.json 加载配置。"""
    if _manager is None:
        init_logging(config=config)
    return _get_manager().get_logger(name)


def cleanup_logs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行日志清理，返回清理报告。"""
    if _manager is None:
        init_logging(config=config)
    return _get_manager().cleanup()
