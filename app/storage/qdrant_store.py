from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance
)
from app.core.config import settings
import uuid
import logging

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.QDRANT_COLLECTION
        self.text_vector_size = settings.TEXT_EMBEDDING_DIM
        self.image_vector_size = settings.IMAGE_EMBEDDING_DIM

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
                vectors_config={
                    settings.TEXT_VECTOR_NAME: VectorParams(
                        size=self.text_vector_size,
                        distance=Distance.COSINE,
                    ),
                    settings.IMAGE_VECTOR_NAME: VectorParams(
                        size=self.image_vector_size,
                        distance=Distance.COSINE,
                    ),
                },
            )
        else:
            logger.info(f"Qdrant collection exists: {self.collection_name}")

    # ---------------------------
    # Write
    # ---------------------------

    def upsert_memory(
        self,
        text_embedding: list[float] | None,
        image_embedding: list[float] | None,
        payload: dict,
    ):
        vectors = {}

        if text_embedding is not None:
            vectors[settings.TEXT_VECTOR_NAME] = text_embedding

        if image_embedding is not None:
            vectors[settings.IMAGE_VECTOR_NAME] = image_embedding

        # point = PointStruct(
        #     id=str(uuid.uuid4()),
        #     vector=vectors,
        #     payload=payload,
        # )
        
        print(vectors)
        quit()
        
        point = {
            "id": str(uuid.uuid4()),
            "vector": vectors,
            "payload": payload,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    # ---------------------------
    # Read
    # ---------------------------

    def search_text(self, query_embedding: list[float], top_k: int = 3):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=(settings.TEXT_VECTOR_NAME, query_embedding),
            limit=top_k,
        )

    def search_image(self, query_embedding: list[float], top_k: int = 3):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=(settings.IMAGE_VECTOR_NAME, query_embedding),
            limit=top_k,
        )

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


# Singleton
qdrant_store = QdrantStore()