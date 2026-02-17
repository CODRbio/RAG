#!/usr/bin/env python
"""
LLM Manager 演示脚本

演示功能：
1. 加载配置
2. 列出可用 providers
3. dry_run 模式测试
4. 实际调用测试（可选）
5. 日志清理
"""

import sys
import argparse

sys.path.insert(0, ".")

from src.log import get_logger
from src.llm import LLMManager, RawLogStore

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LLM Manager 演示")
    parser.add_argument("--config", default="config/rag_config.json", help="配置文件路径")
    parser.add_argument("--provider", type=str, help="指定 provider（默认使用 config.default）")
    parser.add_argument("--model", type=str, help="指定模型（覆盖默认）")
    parser.add_argument("--real", action="store_true", help="实际调用 API（否则仅 dry_run）")
    parser.add_argument("--cleanup", action="store_true", help="执行日志清理")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("LLM Manager 演示")
    logger.info("=" * 60)

    # 1. 加载配置
    logger.info("[1] 加载配置...")
    manager = LLMManager.from_json(args.config)
    logger.info("配置文件: %s, 默认 provider: %s, dry_run: %s",
                args.config, manager.config.default, manager.config.dry_run)

    # 2. 列出 providers
    logger.info("[2] 可用 providers:")
    for name in manager.get_provider_names():
        available = manager.is_available(name)
        pcfg = manager.config.providers[name]
        status = "✓" if available else "✗"
        logger.info("%s %s, default_model: %s, models: %s",
                    status, name, pcfg.default_model, list(pcfg.models.keys()))
        if pcfg.params:
            logger.info("  params: %s", list(pcfg.params.keys()))

    # 3. 模型解析演示
    provider_name = args.provider or manager.config.default
    logger.info("[3] 模型解析演示 (provider=%s)", provider_name)

    resolved_default = manager.resolve_model(provider_name)
    logger.info("默认模型: %s", resolved_default)

    if args.model:
        resolved_custom = manager.resolve_model(provider_name, args.model)
        logger.info("指定模型 '%s' -> %s", args.model, resolved_custom)

    # 4. 测试调用
    logger.info("[4] 测试调用 (provider=%s)", provider_name)

    # 强制 dry_run 除非指定 --real
    if not args.real:
        manager.config.dry_run = True
        logger.info("(dry_run 模式)")

    try:
        client = manager.get_client(provider_name)
        messages = [
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "用一句话介绍深海热液喷口。"},
        ]
        
        resp = client.chat(messages, model=args.model)
        
        print(f"\n  provider: {resp['provider']}")
        print(f"  model: {resp['model']}")
        print(f"  final_text: {resp['final_text'][:200]}...")
        print(f"  reasoning_text: {resp['reasoning_text'][:100] if resp['reasoning_text'] else None}")
        print(f"  latency_ms: {resp['meta']['latency_ms']}")
        print(f"  usage: {resp['meta']['usage']}")
        
    except ValueError as e:
        logger.warning("[SKIP] %s", e)
    except Exception as e:
        logger.error("[ERROR] %s", e)

    # 5. 日志清理
    if args.cleanup:
        logger.info("[5] 日志清理")
        report = manager.cleanup_logs()
        logger.info("按时间删除: %s, 按大小删除: %s, 剩余大小: %.2f MB",
                    report['deleted_by_age'], report['deleted_by_size'], report['remaining_mb'])

    logger.info("=" * 60)
    logger.info("演示完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
