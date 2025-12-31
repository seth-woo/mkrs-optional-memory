from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter
)
from app.core.config import settings
import uuid
import logging

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL  # e.g. http://localhost:6333
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = settings.EMBEDDING_DIM

        self._ensure_collection()

    # ---------------------------
    # Collection management
    # ---------------------------

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if self.collection_name not in names:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Qdrant collection exists: {self.collection_name}")

    # ---------------------------
    # Write
    # ---------------------------

    def upsert_embedding(self, embedding: list[float], payload: dict):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=payload
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    # ---------------------------
    # Read
    # ---------------------------

    def similarity_search(self, query_embedding: list[float], top_k: int = 3):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return results

    # ---------------------------
    # Maintenance
    # ---------------------------

    def clear(self):
        logger.warning("Clearing Qdrant memory")
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count
