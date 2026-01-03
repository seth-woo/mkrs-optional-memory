from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # --------------------
    # Project
    # --------------------
    PROJECT_NAME = "MKRS"

    # --------------------
    # Device
    # --------------------
    USE_CUDA = os.getenv("USE_CUDA", "false").lower() == "true"
    DEVICE = "cuda" if USE_CUDA else "cpu"

    # --------------------
    # Vision-Language Model
    # --------------------
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

    # --------------------
    # Qdrant (Mode 2 Memory)
    # --------------------
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mkrs_memory")

    # --------------------
    # Embeddings
    # --------------------
    # EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))
    
    TEXT_EMBEDDING_DIM = 768
    IMAGE_EMBEDDING_DIM = 512

    TEXT_VECTOR_NAME = "text"
    IMAGE_VECTOR_NAME = "image"

settings = Settings()