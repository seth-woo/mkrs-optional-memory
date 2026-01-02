import torch
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        logger.info(f"Loading text embedder: {self.model_name} on {self.device}")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        logger.info("Text embedder loaded successfully")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()


# Singleton (shared across app)
text_embedder = TextEmbedder()