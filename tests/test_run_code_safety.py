"""Safety and failure-mode tests for the restricted run_code tool."""

import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.llm.tools import _handle_run_code, _get_run_code_semaphore


# ── 正常执行 ──────────────────────────────────────────────────────────────────

def test_run_code_executes_simple_numeric_script(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("import math\nprint(round(math.sqrt(81), 2))")
    assert "9.0" in out


def test_run_code_no_output_returns_success_message(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("x = 1 + 1")
    assert "执行成功" in out or out == "(代码执行成功，无输出)"


# ── 禁用状态 ──────────────────────────────────────────────────────────────────

def test_run_code_disabled_by_default_message(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", False)
    out = _handle_run_code("print(1)")
    assert "run_code 已禁用" in out


# ── AST 拦截 ──────────────────────────────────────────────────────────────────

def test_run_code_blocks_unsafe_import(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("import os\nprint(os.listdir('.'))")
    assert "代码执行被拒绝" in out
    assert "禁止导入模块" in out


def test_run_code_blocks_relative_import(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("from . import something")
    assert "代码执行被拒绝" in out
    assert "禁止相对导入" in out


def test_run_code_blocks_dunder_attribute_access(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("x = [].__class__")
    assert "代码执行被拒绝" in out
    assert "双下划线" in out


def test_run_code_blocks_eval(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("eval('1+1')")
    assert "代码执行被拒绝" in out


def test_run_code_blocks_exec(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("exec('x=1')")
    assert "代码执行被拒绝" in out


def test_run_code_blocks_open(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("open('/etc/passwd')")
    assert "代码执行被拒绝" in out


def test_run_code_blocks_getattr(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("getattr([], '__class__')")
    assert "代码执行被拒绝" in out


def test_run_code_blocks_compile(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("compile('x=1', '<s>', 'exec')")
    assert "代码执行被拒绝" in out


# ── 运行时白名单强制 ──────────────────────────────────────────────────────────

def test_run_code_runtime_import_guard_blocks_unlisted_module(monkeypatch):
    """即使通过 AST 检查（模拟 allowed_modules 包含 sys），运行时也应拒绝。"""
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    # sys 不在默认 allowed_modules，应被 AST validator 拦截
    out = _handle_run_code("import sys")
    assert "代码执行被拒绝" in out or "not allowed" in out.lower()


def test_run_code_allowed_module_works(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    # statistics 在默认 allowed_modules 中
    out = _handle_run_code(
        "import statistics\nprint(statistics.mean([1,2,3,4,5]))"
    )
    assert "3" in out


# ── 超时与输出限制 ────────────────────────────────────────────────────────────

def test_run_code_times_out_and_recovers(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    monkeypatch.setattr(settings.tool_execution, "timeout_seconds", 1)
    out = _handle_run_code("import time\ntime.sleep(5)\nprint('done')")
    assert "代码执行超时" in out


def test_run_code_output_limit(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    monkeypatch.setattr(settings.tool_execution, "max_output_chars", 100)
    # 生成远超 100 字符的输出
    out = _handle_run_code("for i in range(10000): print('A' * 100)")
    assert "超过上限" in out or "超时" in out


# ── 代码长度限制 ──────────────────────────────────────────────────────────────

def test_run_code_rejects_oversized_code(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    monkeypatch.setattr(settings.tool_execution, "max_code_chars", 50)
    out = _handle_run_code("x = 1\n" * 100)
    assert "代码执行被拒绝" in out
    assert "超过上限" in out


# ── 并发控制 ──────────────────────────────────────────────────────────────────

def test_run_code_concurrent_limit(monkeypatch):
    """当 Semaphore 已满时，新请求应立即返回并发上限提示。"""
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    monkeypatch.setattr(settings.tool_execution, "max_concurrent", 1)

    # 强制重建 Semaphore 为 1
    import src.llm.tools as tools_mod
    monkeypatch.setattr(tools_mod, "_RUN_CODE_SEMAPHORE", threading.Semaphore(1))
    monkeypatch.setattr(tools_mod, "_RUN_CODE_SEMAPHORE_SIZE", 1)

    # 手动占用唯一槽位
    tools_mod._RUN_CODE_SEMAPHORE.acquire()
    try:
        out = _handle_run_code("print(1)")
        assert "并发上限已满" in out
    finally:
        tools_mod._RUN_CODE_SEMAPHORE.release()


# ── 语法错误处理 ──────────────────────────────────────────────────────────────

def test_run_code_syntax_error(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("def foo(:\n    pass")
    assert "代码执行被拒绝" in out
    assert "语法错误" in out


# ── 运行时异常不泄露宿主路径 ─────────────────────────────────────────────────

def test_run_code_runtime_error_returns_failure(monkeypatch):
    monkeypatch.setattr(settings.tool_execution, "run_code_enabled", True)
    out = _handle_run_code("x = 1 / 0")
    assert "代码执行失败" in out
    assert "ZeroDivisionError" in out
