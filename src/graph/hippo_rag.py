"""
HippoRAG 风格知识图谱增强检索

核心思想（受海马体索引理论启发）：
1. 从文档中抽取实体和关系，构建知识图谱
2. 检索时先通过向量检索找到种子节点
3. 使用 Personalized PageRank 在图上扩展
4. 融合向量检索和图检索结果

参考：https://github.com/OSU-NLP-Group/HippoRAG
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import networkx as nx

from src.chunking.chunker import ChunkConfig, chunk_blocks
from src.log import get_logger


@dataclass
class Entity:
    """实体"""
    name: str
    type: str  # SPECIES, LOCATION, PHENOMENON, METHOD, SUBSTANCE 等
    mentions: List[str] = field(default_factory=list)  # 出现的 chunk_ids


@dataclass
class Relation:
    """关系"""
    source: str
    target: str
    relation_type: str  # 如 FOUND_IN, PRODUCES, LIVES_AT 等
    chunk_id: str


class HippoRAG:
    """
    HippoRAG 知识图谱检索增强

    使用 NetworkX 构建图谱，支持：
    - 实体抽取（基于规则或 LLM）
    - 关系抽取
    - Personalized PageRank 检索扩展
    """

    def __init__(self, graph_path: Optional[Path] = None):
        self.logger = get_logger(__name__)
        self.G = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self.entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        self.graph_path = graph_path

        if graph_path and graph_path.exists():
            self.load(graph_path)

    # ========== 实体抽取 ==========

    def extract_entities_rule_based(self, text: str, chunk_id: str) -> List[Entity]:
        """
        基于规则的实体抽取（适用于深海科研领域）

        可扩展为 LLM 抽取，但规则方法更快且可控
        """
        entities = []

        # 深海相关实体模式
        patterns = {
            "LOCATION": [
                r"(马里亚纳海沟|冲绳海槽|南海|东太平洋|大西洋中脊)",
                r"(热液喷口|冷泉|深海平原|海山|海沟)",
                r"(hydrothermal vent|cold seep|deep sea|trench)",
            ],
            "SPECIES": [
                r"(管虫|贻贝|蟹类|虾类|细菌|古菌)",
                r"(tube worm|mussel|crab|shrimp|bacteria|archaea)",
                r"([A-Z][a-z]+ [a-z]+)",  # 拉丁学名
            ],
            "PHENOMENON": [
                r"(化能合成|光合作用|共生|甲烷渗漏|热液活动)",
                r"(chemosynthesis|symbiosis|methane seep)",
            ],
            "SUBSTANCE": [
                r"(硫化氢|甲烷|二氧化碳|氧气|硫化物)",
                r"(H2S|CH4|CO2|sulfide|methane)",
            ],
            "METHOD": [
                r"(ROV|AUV|深潜器|采样器|声呐)",
                r"(remotely operated vehicle|autonomous underwater vehicle)",
            ],
        }

        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match if isinstance(match, str) else match[0]
                    name = name.strip().lower()
                    if len(name) > 2:  # 过滤太短的匹配
                        entity = Entity(name=name, type=entity_type, mentions=[chunk_id])
                        entities.append(entity)

        return entities

    def extract_entities_llm(self, text: str, chunk_id: str, llm_client=None) -> List[Entity]:
        """
        基于 LLM 的实体抽取（更准确但更慢）

        需要传入 LLM client，返回结构化实体列表
        """
        if llm_client is None:
            return self.extract_entities_rule_based(text, chunk_id)

        prompt = f"""从以下深海科研文本中抽取实体，返回 JSON 格式：

文本：
{text[:2000]}

请抽取以下类型的实体：
- LOCATION: 地理位置、海域、地质构造
- SPECIES: 物种名称（包括拉丁学名）
- PHENOMENON: 自然现象、生物过程
- SUBSTANCE: 化学物质、元素
- METHOD: 研究方法、设备

返回格式：
[{{"name": "实体名", "type": "类型"}}]

仅返回 JSON，不要其他内容："""

        try:
            # 这里假设使用 Claude API
            response = llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            result = json.loads(response.content[0].text)
            return [
                Entity(name=e["name"].lower(), type=e["type"], mentions=[chunk_id])
                for e in result
            ]
        except Exception:
            return self.extract_entities_rule_based(text, chunk_id)

    # ========== 图谱构建 ==========

    def add_chunk(self, chunk_id: str, content: str, paper_id: str, use_llm: bool = False, llm_client=None):
        """
        处理单个 chunk，抽取实体并添加到图谱
        """
        # 抽取实体
        if use_llm and llm_client:
            entities = self.extract_entities_llm(content, chunk_id, llm_client)
        else:
            entities = self.extract_entities_rule_based(content, chunk_id)

        # 添加实体节点
        for entity in entities:
            if entity.name not in self.entities:
                self.entities[entity.name] = entity
                self.G.add_node(entity.name, type=entity.type)
            else:
                self.entities[entity.name].mentions.append(chunk_id)

            self.chunk_to_entities[chunk_id].add(entity.name)
            self.entity_to_chunks[entity.name].add(chunk_id)

        # 添加 chunk 节点
        self.G.add_node(chunk_id, type="CHUNK", paper_id=paper_id)

        # 实体 -> chunk 边
        for entity in entities:
            self.G.add_edge(entity.name, chunk_id, relation="MENTIONED_IN")

        # 同一 chunk 内的实体之间建立共现关系
        entity_names = [e.name for e in entities]
        for i, e1 in enumerate(entity_names):
            for e2 in entity_names[i + 1:]:
                if self.G.has_edge(e1, e2):
                    self.G[e1][e2]["weight"] = self.G[e1][e2].get("weight", 1) + 1
                else:
                    self.G.add_edge(e1, e2, relation="CO_OCCURS", weight=1)
                if self.G.has_edge(e2, e1):
                    self.G[e2][e1]["weight"] = self.G[e2][e1].get("weight", 1) + 1
                else:
                    self.G.add_edge(e2, e1, relation="CO_OCCURS", weight=1)

    def build_from_parsed_docs(
        self,
        parsed_dir: Path,
        use_llm: bool = False,
        llm_client=None,
        chunk_config: Optional[ChunkConfig] = None,
    ):
        """
        从解析后的文档构建图谱。递归扫描 parsed_dir 下的 enriched.json，
        使用与 03_index 一致的 content_flow + chunk_blocks 分块，保证 chunk_id 与 Milvus 一致。
        """
        json_files = list(parsed_dir.rglob("enriched.json"))
        self.logger.info(f"构建知识图谱，共 {len(json_files)} 个文档...")

        cfg = chunk_config or ChunkConfig()

        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)

            doc_id = doc.get("doc_id", json_path.parent.name)
            paper_id = doc_id
            content_flow = doc.get("content_flow", [])

            chunks = chunk_blocks(content_flow, doc_id=doc_id, config=cfg)
            for c in chunks:
                if c.text.strip():
                    self.add_chunk(c.chunk_id, c.text, paper_id, use_llm, llm_client)

        self.logger.info(f"图谱构建完成: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")

    # ========== 检索增强 ==========

    def get_seed_entities(self, query: str) -> List[str]:
        """
        从查询中抽取实体作为种子节点
        """
        entities = self.extract_entities_rule_based(query, "query")
        seed_names = [e.name for e in entities if e.name in self.entities]
        return seed_names

    def personalized_pagerank(
        self,
        seed_entities: List[str],
        alpha: float = 0.85,
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Personalized PageRank

        从种子实体出发，在图上扩展找到相关节点
        """
        if not seed_entities:
            return []

        # 构建 personalization 向量
        personalization = {node: 0.0 for node in self.G.nodes()}
        for entity in seed_entities:
            if entity in personalization:
                personalization[entity] = 1.0 / len(seed_entities)

        # 运行 PPR
        try:
            scores = nx.pagerank(
                self.G,
                alpha=alpha,
                personalization=personalization,
                max_iter=100
            )
        except nx.PowerIterationFailedConvergence:
            # 如果不收敛，返回空
            return []

        # 过滤出 chunk 节点并排序
        chunk_scores = [
            (node, score)
            for node, score in scores.items()
            if self.G.nodes[node].get("type") == "CHUNK"
        ]
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        return chunk_scores[:top_k]

    def retrieve_with_graph(
        self,
        query: str,
        vector_hits: List[Dict],
        top_k: int = 10,
        graph_weight: float = 0.3
    ) -> List[Dict]:
        """
        融合向量检索和图检索结果

        Args:
            query: 查询文本
            vector_hits: 向量检索结果（需包含 chunk_id）
            top_k: 返回数量
            graph_weight: 图检索权重（0-1）

        Returns:
            融合后的检索结果
        """
        # 1. 从查询抽取种子实体
        seed_entities = self.get_seed_entities(query)

        # 2. 如果有种子实体，进行 PPR 扩展
        graph_scores = {}
        if seed_entities:
            ppr_results = self.personalized_pagerank(seed_entities, top_k=top_k * 2)
            max_score = ppr_results[0][1] if ppr_results else 1.0
            for chunk_id, score in ppr_results:
                graph_scores[chunk_id] = score / max_score  # 归一化到 0-1

        # 3. 融合向量检索分数和图检索分数
        fused_results = []
        for hit in vector_hits:
            chunk_id = hit.get("metadata", {}).get("chunk_id", "")
            vector_score = hit.get("score", 0)

            # 归一化向量分数（假设已经是 0-1 范围）
            graph_score = graph_scores.get(chunk_id, 0)

            # 加权融合
            fused_score = (1 - graph_weight) * vector_score + graph_weight * graph_score

            fused_hit = hit.copy()
            fused_hit["score"] = fused_score
            fused_hit["vector_score"] = vector_score
            fused_hit["graph_score"] = graph_score
            fused_results.append(fused_hit)

        # 4. 添加仅图检索命中的结果
        vector_chunk_ids = {h.get("metadata", {}).get("chunk_id", "") for h in vector_hits}
        for chunk_id, score in graph_scores.items():
            if chunk_id not in vector_chunk_ids:
                paper_id = self.G.nodes[chunk_id].get("paper_id", "")
                fused_results.append({
                    "chunk_id": chunk_id,
                    "metadata": {"chunk_id": chunk_id, "paper_id": paper_id},
                    "score": graph_weight * score,
                    "vector_score": 0,
                    "graph_score": score,
                    "source": "graph_only"
                })

        # 5. 排序并返回
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return fused_results[:top_k]

    # ========== 持久化 ==========

    def save(self, path: Path):
        """保存图谱到文件"""
        data = {
            "nodes": list(self.G.nodes(data=True)),
            "edges": list(self.G.edges(data=True)),
            "entities": {k: {"name": v.name, "type": v.type, "mentions": v.mentions}
                        for k, v in self.entities.items()},
            "chunk_to_entities": {k: list(v) for k, v in self.chunk_to_entities.items()},
            "entity_to_chunks": {k: list(v) for k, v in self.entity_to_chunks.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"图谱已保存: {path}")

    def load(self, path: Path):
        """从文件加载图谱"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.G = nx.DiGraph()
        for node, attrs in data["nodes"]:
            self.G.add_node(node, **attrs)
        for u, v, attrs in data["edges"]:
            self.G.add_edge(u, v, **attrs)

        self.entities = {
            k: Entity(name=v["name"], type=v["type"], mentions=v["mentions"])
            for k, v in data["entities"].items()
        }
        self.chunk_to_entities = defaultdict(set, {k: set(v) for k, v in data["chunk_to_entities"].items()})
        self.entity_to_chunks = defaultdict(set, {k: set(v) for k, v in data["entity_to_chunks"].items()})

        self.logger.info(f"图谱已加载: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")

    # ========== 统计 ==========

    def stats(self) -> Dict:
        """返回图谱统计信息"""
        entity_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") != "CHUNK"]
        chunk_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") == "CHUNK"]

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "entity_count": len(entity_nodes),
            "chunk_count": len(chunk_nodes),
            "entity_types": dict(defaultdict(int, {
                d.get("type"): 1 for _, d in self.G.nodes(data=True) if d.get("type") != "CHUNK"
            })),
        }


# 全局实例（懒加载）
_hippo_rag: Optional[HippoRAG] = None


def get_hippo_rag(graph_path: Optional[Path] = None) -> HippoRAG:
    """获取 HippoRAG 单例"""
    global _hippo_rag
    if _hippo_rag is None:
        _hippo_rag = HippoRAG(graph_path)
    return _hippo_rag
