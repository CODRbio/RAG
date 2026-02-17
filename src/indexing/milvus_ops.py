"""
Milvus 操作封装
Mac 和服务器使用完全相同的代码
"""

from pymilvus import MilvusClient, DataType
from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)


class MilvusOps:
    """Milvus 操作类"""

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=settings.milvus.uri)
        return self._client

    def create_collection(self, name: str, recreate: bool = False, schema_version: str = "v1"):
        """
        创建 Collection。
        schema_version: "v1" = auto_id + id 主键（兼容旧脚本）；"v2" = chunk_id 主键，支持 upsert。
        """
        if self.client.has_collection(name):
            if recreate:
                self.client.drop_collection(name)
            else:
                logger.info(f"  Collection '{name}' 已存在")
                return

        if schema_version == "v2":
            self._create_collection_v2(name)
        else:
            self._create_collection_v1(name)

    def _create_collection_v1(self, name: str):
        """v1: auto_id + id 主键（兼容原有 insert）"""
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("raw_content", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("paper_id", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=256)
        schema.add_field("domain", DataType.VARCHAR, max_length=64)
        schema.add_field("content_type", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=128)
        schema.add_field("section_path", DataType.VARCHAR, max_length=65535)
        schema.add_field("page", DataType.INT32)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", **settings.index.params)
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self.client.create_collection(collection_name=name, schema=schema, index_params=index_params)
        logger.info(f"  [OK] Collection '{name}' 创建成功 (v1)")

    def _create_collection_v2(self, name: str):
        """v2: chunk_id 主键，支持 upsert，重复跑不产生重复数据"""
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("raw_content", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("paper_id", DataType.VARCHAR, max_length=512)
        schema.add_field("domain", DataType.VARCHAR, max_length=64)
        schema.add_field("content_type", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=128)
        schema.add_field("section_path", DataType.VARCHAR, max_length=65535)
        schema.add_field("page", DataType.INT32)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", **settings.index.params)
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self.client.create_collection(collection_name=name, schema=schema, index_params=index_params)
        logger.info(f"  [OK] Collection '{name}' 创建成功 (v2, chunk_id PK)")

    def init_all_collections(self, recreate: bool = False):
        """初始化所有子库"""
        logger.info("初始化 Collections...")
        for name in settings.collection.all():
            self.create_collection(name, recreate)

    def insert(self, collection: str, data: list):
        return self.client.insert(collection_name=collection, data=data)

    def upsert(self, collection: str, data: list):
        """Upsert 数据（需 collection 为 chunk_id 主键，即 schema_version='v2'）"""
        if not data:
            return
        return self.client.upsert(collection_name=collection, data=data)

    def search(self, collection: str, **kwargs):
        return self.client.search(collection_name=collection, **kwargs)

    def hybrid_search(self, collection: str, **kwargs):
        return self.client.hybrid_search(collection_name=collection, **kwargs)

    def query(self, collection: str, **kwargs):
        return self.client.query(collection_name=collection, **kwargs)

    def count(self, collection: str) -> int:
        stats = self.client.get_collection_stats(collection)
        return stats.get("row_count", 0)


# 全局实例
milvus = MilvusOps()
