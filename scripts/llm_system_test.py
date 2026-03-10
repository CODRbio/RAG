#!/usr/bin/env python3
"""
LLM 系统测试脚本：用统一问题调用配置中的 LLM，输出「提问 + 回复 + 元信息」便于人工核对与反馈。

用法:
  # 使用默认问题（未来人类思维与 AI）
  conda run -n deepsea-rag python scripts/llm_system_test.py

  # 指定问题
  conda run -n deepsea-rag python scripts/llm_system_test.py --question "你的问题"

  # 指定单个 provider（与 config 中 default 一致时可省略）
  conda run -n deepsea-rag python scripts/llm_system_test.py --provider deepseek

  # 对所有「已配置 API key」的 provider 各跑一遍，便于横向对比
  conda run -n deepsea-rag python scripts/llm_system_test.py --all

  # 结果同时写入文件
  conda run -n deepsea-rag python scripts/llm_system_test.py --all --out scripts/llm_system_test_output.txt

  # 仅打印题目与将使用的 provider，不请求 API（便于检查环境）
  conda run -n deepsea-rag python scripts/llm_system_test.py --all --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_QUESTION = "未来人类的思维会不会因为AI的使用而退化甚至终结？"
DEFAULT_OUTPUT_FILE = ROOT / "scripts" / "llm_system_test_output.txt"


def _run_one(
    manager,
    provider: str,
    question: str,
    timeout_seconds: int = 120,
) -> dict:
    """对单个 provider 发起一次 chat，返回 {success, provider, model, text, meta, error}"""
    out = {
        "success": False,
        "provider": provider,
        "model": None,
        "text": None,
        "meta": None,
        "error": None,
    }
    try:
        client = manager.get_client(provider)
        resolved_model = getattr(client, "_resolve_model", lambda m: None)(None) or getattr(
            client.config, "default_model", ""
        )
        out["model"] = resolved_model

        messages = [{"role": "user", "content": question}]
        resp = client.chat(messages, model=None, timeout_seconds=timeout_seconds)

        out["success"] = True
        out["text"] = (resp.get("final_text") or "").strip()
        usage = (resp.get("meta") or {}).get("usage") or {}
        latency = (resp.get("meta") or {}).get("latency_ms")
        out["meta"] = {
            "usage": usage,
            "latency_ms": latency,
            "citations": resp.get("citations"),
            "search_results": resp.get("search_results"),
        }
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def _format_one(result: dict, question: str) -> str:
    lines = [
        "",
        "=" * 60,
        f"Provider: {result['provider']}",
        f"Model:    {result.get('model') or '-'}",
        "=" * 60,
        "【提问】",
        question,
        "",
        "【回复】",
    ]
    if result["success"]:
        lines.append(result.get("text") or "(无文本)")
        meta = result.get("meta") or {}
        if meta.get("usage"):
            lines.append("")
            lines.append("【用量】 " + str(meta["usage"]))
        if meta.get("latency_ms") is not None:
            lines.append(f"【耗时】 {meta['latency_ms']} ms")
        if meta.get("citations"):
            lines.append("【引用】 " + str(meta["citations"][:5]) + (" ..." if len(meta["citations"]) > 5 else ""))
    else:
        lines.append(f"【错误】 {result.get('error')}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM 系统测试：统一提问，输出提问与回复便于人工反馈",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="向模型提出的问题（默认：关于人类思维与 AI 的题目）",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="使用的 provider 名称（默认使用 config 中的 default）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="对每个已配置 API key 的 provider 各跑一次",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="将完整输出追加写入该文件（默认不写文件）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="单次请求超时秒数（默认 120）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印题目与 provider 列表，不发起请求",
    )
    args = parser.parse_args()

    config_path = ROOT / "config" / "rag_config.json"
    try:
        from src.llm.llm_manager import get_manager
        manager = get_manager(str(config_path))
    except Exception as e:
        print("加载 LLM 配置失败:", e, file=sys.stderr)
        return 1

    question = (args.question or "").strip() or DEFAULT_QUESTION
    all_output: list[str] = []

    def emit(s: str):
        print(s)
        all_output.append(s)

    emit("LLM 系统测试")
    emit("提问（共 1 题）：")
    emit(question)
    emit("")

    if not args.all and not args.dry_run:
        provider = args.provider or manager.config.default
        emit(f"当前仅使用默认 provider: {provider}（要对所有已配置模型各跑一遍请加 --all）")
        emit("")

    if args.dry_run:
        default_provider = manager.config.default
        emit(f"默认 provider: {default_provider}")
        emit("所有 provider: " + ", ".join(manager.get_provider_names()))
        available = [p for p in manager.get_provider_names() if manager.is_available(p)]
        emit("已配置 API key 的 provider: " + (", ".join(available) if available else "(无)"))
        emit("")
        if args.out is not None:
            args.out.write_text("\n".join(all_output), encoding="utf-8")
            print(f"已写入: {args.out}", file=sys.stderr)
        return 0

    if args.all:
        providers = [p for p in manager.get_provider_names() if manager.is_available(p)]
        if not providers:
            emit("未找到任何已配置 API key 的 provider，请检查 config 或环境变量。")
            if args.out is not None:
                args.out.write_text("\n".join(all_output), encoding="utf-8")
            return 1
        emit(f"将对以下 {len(providers)} 个 provider 分别请求: {', '.join(providers)}")
        results = []
        for p in providers:
            results.append(_run_one(manager, p, question, timeout_seconds=args.timeout))
        for r in results:
            emit(_format_one(r, question))
        # 简短汇总
        ok = sum(1 for r in results if r["success"])
        emit("")
        emit(f"汇总: {ok}/{len(results)} 成功")
    else:
        provider = args.provider or manager.config.default
        if not manager.is_available(provider):
            emit(f"Provider '{provider}' 不可用或未配置 API key。可用: {manager.get_provider_names()}")
            if args.out is not None:
                args.out.write_text("\n".join(all_output), encoding="utf-8")
            return 1
        result = _run_one(manager, provider, question, timeout_seconds=args.timeout)
        emit(_format_one(result, question))
        if not result["success"]:
            if args.out is not None:
                args.out.write_text("\n".join(all_output), encoding="utf-8")
            return 1

    if args.out is not None:
        args.out.write_text("\n".join(all_output), encoding="utf-8")
        print(f"\n完整输出已写入: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
