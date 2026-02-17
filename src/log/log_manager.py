"""
日志管理模块：分级日志、按运行实例命名、自动清理。
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# 默认配置
DEFAULT_LEVEL = "INFO"
DEFAULT_MAX_SIZE_MB = 100
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_MIN_KEEP_MB = 20
DEFAULT_CONSOLE_OUTPUT = True
LOG_DIR_NAME = "app"


class LogManager:
    """
    统一日志管理：分级（DEBUG/INFO/WARNING/ERROR）、按运行实例命名文件、
    控制台+文件双输出、按配置自动清理（上限 100MB，低于 20MB 不删）。
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        base = Path(__file__).resolve().parent.parent.parent
        self.log_dir = Path(config.get("log_dir")) if config.get("log_dir") else base / "logs" / LOG_DIR_NAME
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_mb = int(config.get("max_size_mb", DEFAULT_MAX_SIZE_MB))
        self.max_age_days = int(config.get("max_age_days", DEFAULT_MAX_AGE_DAYS))
        self.min_keep_mb = int(config.get("min_keep_mb", DEFAULT_MIN_KEEP_MB))
        self.console_output = config.get("console_output", DEFAULT_CONSOLE_OUTPUT)
        level_name = (config.get("level") or DEFAULT_LEVEL).upper()
        self.level = getattr(logging, level_name, logging.INFO)

        self._run_log_path: Path | None = None
        self._formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._formatter.default_msec_format = "%s.%03d"

    def _run_file(self) -> Path:
        """当前运行的日志文件路径（按启动时间命名，进程内复用）."""
        if self._run_log_path is None:
            self._run_log_path = self.log_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        return self._run_log_path

    def get_logger(self, name: str) -> logging.Logger:
        """获取具名 logger，自动绑定当前运行日志文件与控制台."""
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger

        logger.setLevel(self.level)
        logger.propagate = False

        if self.console_output:
            ch = logging.StreamHandler()
            ch.setLevel(self.level)
            ch.setFormatter(self._formatter)
            logger.addHandler(ch)

        fh = logging.FileHandler(self._run_file(), encoding="utf-8")
        fh.setLevel(self.level)
        fh.setFormatter(self._formatter)
        logger.addHandler(fh)

        return logger

    def cleanup(self) -> dict[str, Any]:
        """
        按配置清理日志文件。
        - 总大小 < min_keep_mb 时不删除。
        - 先删超过 max_age_days 的文件，再按最旧优先删至不超过 max_size_mb。
        """
        report: dict[str, Any] = {"deleted_by_age": [], "deleted_by_size": [], "remaining_mb": 0.0}
        if not self.log_dir.exists():
            return report

        log_files = sorted(
            (f for f in self.log_dir.iterdir() if f.is_file() and f.suffix == ".log"),
            key=lambda p: p.stat().st_mtime,
        )
        min_bytes = self.min_keep_mb * 1024 * 1024
        total = sum(f.stat().st_size for f in log_files)
        if total < min_bytes:
            report["remaining_mb"] = total / (1024 * 1024)
            return report

        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        remaining: list[Path] = []
        for f in log_files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                report["deleted_by_age"].append(f.name)
                f.unlink()
            else:
                remaining.append(f)

        max_bytes = self.max_size_mb * 1024 * 1024
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


# 模块级单例，便于 get_logger 使用
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


def init_logging(config: dict[str, Any] | None = None, config_path: str | Path | None = None) -> LogManager:
    """初始化日志（可选从配置文件加载）. 显式调用时可传入 config 或 config_path."""
    cfg = config
    if cfg is None and config_path is not None:
        path = Path(config_path)
        cfg = _load_logging_config(path)
    elif cfg is None:
        default_path = Path(__file__).resolve().parent.parent.parent / "config" / "rag_config.json"
        cfg = _load_logging_config(default_path)
    global _manager
    _manager = LogManager(cfg)
    return _manager


def get_logger(name: str, config: dict[str, Any] | None = None) -> logging.Logger:
    """获取 logger。若尚未初始化则用 config 或 rag_config.json 的 logging 段初始化."""
    if _manager is None:
        init_logging(config=config)
    return _get_manager().get_logger(name)


def cleanup_logs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行日志清理，返回清理报告。若未初始化则用默认或 config 初始化 LogManager 再清理."""
    if _manager is None:
        init_logging(config=config)
    return _get_manager().cleanup()
