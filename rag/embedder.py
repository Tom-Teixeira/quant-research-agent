"""嵌入层：中文金融文本向量化（RAG 检索的向量来源）。

选型说明：
  - BAAI/bge-small-zh-v1.5：512 维，中文金融场景主流轻量模型，与 pgvector HNSW 配合
    在 10万+ 规模下检索毫秒级；
  - fastembed（ONNX Runtime）实现，避免安装 PyTorch（~2GB），模型 ~100MB 首次自动下载；
  - 模型源：优先 ModelScope（中国大陆可达），失败降级 HuggingFace。
"""
from __future__ import annotations

import os

# 中国大陆默认走 HuggingFace 镜像（用户已显式设置 HF_ENDPOINT 时尊重用户配置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_NAME = os.getenv("QUANT_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")


class Embedder:
    """轻量中文嵌入器（fastembed / ONNX）。"""

    def __init__(self, model_name: str = MODEL_NAME):
        from fastembed import TextEmbedding  # 延迟导入，节省启动

        self._model = TextEmbedding(model_name=model_name)
        self.dim = 512  # bge-small-zh-v1.5

    def embed(self, texts: list[str]) -> list[list[float]]:
        """文本 → 向量列表（模型内部自动归一化，配合 cosine 距离）。"""
        if not texts:
            return []
        return [list(v) for v in self._model.embed(texts)]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """单例获取（模型只加载一次）。"""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
