"""
统一 LLM 调用：兼容层

本模块保留 call_llm() 函数签名，内部委托给 src/llm/llm_manager.py。
旧脚本可继续使用，无需修改调用代码。
"""

from typing import Optional

from src.llm.llm_manager import get_manager, LLMManager


def call_llm(
    provider: str,
    system: str,
    user_prompt: str,
    model_override: Optional[str] = None,
    max_tokens: int = 2000,
) -> str:
    """
    按 provider 调用对应 LLM，返回生成文本。

    Args:
        provider: config 中定义的 provider key (如 openai, deepseek, gemini-vision)
        system: system prompt
        user_prompt: user message
        model_override: 若传入则覆盖 config 中的 model
        max_tokens: 最大 token 数

    Returns:
        模型返回的文本；失败时返回以 [ERROR] 开头的字符串。
    """
    try:
        manager = get_manager()
    except Exception as e:
        return f"[ERROR] 加载 LLM 配置失败: {e}"

    # 检查 provider 是否可用
    if not manager.is_available(provider):
        return f"[ERROR] 未配置 {provider} 的 API Key，请在 config/rag_config.json 或环境变量中设置"

    try:
        client = manager.get_client(provider)
    except ValueError as e:
        return f"[ERROR] {e}"

    # 构建消息
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = client.chat(messages, model=model_override, max_tokens=max_tokens)
        return (resp.get("final_text") or "").strip()
    except Exception as e:
        return f"[ERROR] {provider} 调用失败: {e}"
