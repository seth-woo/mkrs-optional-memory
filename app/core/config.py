from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    PROJECT_NAME = "MKRS"
    USE_CUDA = os.getenv("USE_CUDA", "false").lower() == "true"
    DEVICE = "cuda" if USE_CUDA else "cpu"
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B"

settings = Settings()