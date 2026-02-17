#!/usr/bin/env python
"""步骤5: RAG 端到端测试（支持 HippoRAG）"""

import sys
import json
import os
from datetime import datetime

sys.path.insert(0, ".")

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)
from src.indexing.milvus_ops import milvus
from src.retrieval.hybrid_retriever import retriever, RetrievalConfig
from src.generation.llm_client import call_llm


def retrieve(query: str, collection: str, top_k: int = 5, mode: str = "hybrid"):
    """
    检索相关文档

    Args:
        mode: vector / graph / hybrid
    """
    config = RetrievalConfig(mode=mode, top_k=top_k, rerank=True, graph_weight=0.3)
    hits = retriever.retrieve(query, collection, config)

    # 转换为兼容格式
    converted = []
    for hit in hits:
        converted.append(type("Hit", (), {
            "entity": {
                "paper_id": hit.get("metadata", {}).get("paper_id", ""),
                "chunk_id": hit.get("metadata", {}).get("chunk_id", ""),
                "content": hit.get("content", ""),
                "raw_content": hit.get("raw_content", ""),
                "section_path": hit.get("metadata", {}).get("section_path", ""),
                "page": hit.get("metadata", {}).get("page", 0),
            },
            "distance": hit.get("score", 0),
            "graph_score": hit.get("graph_score", 0),
            "vector_score": hit.get("vector_score", 0),
        })())
    return converted


def build_context(hits) -> str:
    """构建上下文"""
    parts = []
    for i, hit in enumerate(hits, 1):
        content = hit.entity.get("raw_content") or hit.entity.get("content", "")
        section = hit.entity.get("section_path", "")
        paper_id = hit.entity.get("paper_id", "")

        parts.append(f"""
【参考文献 {i}】
论文: {paper_id}
章节: {section}
内容:
{content[:1000]}
""")
    return "\n".join(parts)


def generate_answer(
    query: str,
    context: str,
    dry_run: bool = False,
    llm_provider: str = None,
    model_override: str = None,
) -> str:
    """
    生成回答。provider 与 model 可由调用方通过参数指定。
    """
    system_prompt = """你是深海科学研究助手。基于参考文献回答问题。

规则：
1. 用 [来源X] 标注引用
2. 证据不足时说明"根据现有资料无法确定"
3. 不要编造信息"""

    user_prompt = f"""基于以下参考文献回答问题。

{context}

问题: {query}

请给出带引用标注的回答:"""

    if dry_run:
        return f"[DRY_RUN] 问题: {query}\n上下文长度: {len(context)} 字符\n参考文献数: {context.count('【参考文献')}"

    provider = (llm_provider or settings.llm.default).lower()
    
    if not settings.llm.is_available(provider):
        return f"[ERROR] 未配置 {provider} 的 API Key，请在 config/rag_config.json 或环境变量中设置"

    return call_llm(
        provider=provider,
        system=system_prompt,
        user_prompt=user_prompt,
        model_override=model_override,
        max_tokens=2000,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vector", "graph", "hybrid"], default="hybrid",
                       help="检索模式: vector/graph/hybrid")
    parser.add_argument("--collection", type=str, default=None,
                       help="指定 collection 名称（默认用 settings.collection.global_）")
    parser.add_argument("--llm", type=str, default=None,
                       help="LLM 提供方 (如 openai, deepseek, gemini-thinking)，默认用 config/rag_config.json 中的 default")
    parser.add_argument("--model", type=str, default=None,
                       help="本次调用使用的模型名或别名，覆盖 config 中该 provider 的 default_model/models")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings.path.ensure_dirs()

    logger.info("=" * 60)
    logger.info("深海科研知识库 - RAG 测试（HippoRAG）")
    logger.info("=" * 60)

    dry_run = settings.llm.dry_run
    retrieval_mode = args.mode
    llm_provider = args.llm or settings.llm.default
    model_override = args.model

    logger.info("检索模式: %s, LLM: %s%s", retrieval_mode, llm_provider,
                f" (model={model_override})" if model_override else "")
    if dry_run:
        logger.info("LLM_DRY_RUN=true，仅测试检索链路")

    collection_name = args.collection or settings.collection.global_
    test_questions = [
        "深海热液喷口的温度范围是多少？",
        "深海生态系统有哪些特点？",
    ]

    artifact = {
        "run_id": run_id,
        "dry_run": dry_run,
        "retrieval_mode": retrieval_mode,
        "llm_provider": llm_provider,
        "model_override": model_override,
        "results": []
    }

    # 检查 collection
    count = milvus.count(collection_name)
    if count == 0:
        logger.warning("Collection 为空")
        logger.warning("请先运行: python scripts/06_ingest_langgraph.py")
        return

    for question in test_questions:
        logger.info("问题: %s", question)

        result = {"question": question}

        # 1. 检索
        logger.info("[1] 检索...")
        hits = retrieve(question, collection_name, top_k=3, mode=retrieval_mode)
        logger.info("命中: %s 条", len(hits))

        result["sources"] = []
        for i, hit in enumerate(hits, 1):
            paper_id = hit.entity.get("paper_id", "N/A")
            chunk_id = hit.entity.get("chunk_id", "N/A")
            graph_score = getattr(hit, "graph_score", 0)
            vector_score = getattr(hit, "vector_score", 0)
            logger.info("[%s] %s - %s", i, paper_id, chunk_id)
            if retrieval_mode == "hybrid":
                logger.info("vector=%s, graph=%s", f"{vector_score:.3f}", f"{graph_score:.3f}")
            result["sources"].append({
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "graph_score": graph_score,
                "vector_score": vector_score
            })

        # 2. 构建上下文
        context = build_context(hits)

        # 3. 生成回答
        logger.info("[2] 生成回答...")
        answer = generate_answer(
            question, context, dry_run=dry_run,
            llm_provider=llm_provider, model_override=model_override,
        )

        print(f"\n回答:\n{answer}")

        # 4. 检查引用
        has_citation = "[来源" in answer or "[来源1]" in answer or "参考文献" in answer
        result["answer"] = answer
        result["has_citation"] = has_citation
        result["context_length"] = len(context)

        if not dry_run and not has_citation:
            logger.warning("回答中未检测到引用标记")

        artifact["results"].append(result)

    # 保存 artifact
    artifact_path = settings.path.artifacts / f"05_rag_{run_id}.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("RAG 测试完成")
    logger.info("产物已保存: %s", artifact_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
