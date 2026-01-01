import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "openai/clip-vit-base-patch32"

        logger.info(f"Loading image embedder: {self.model_name} on {self.device}")

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        logger.info("Image embedder loaded successfully")

    def embed(self, images: list[Image.Image]) -> list[list[float]]:
        """
        Generate embeddings for a list of PIL Images.
        """
        if not images:
            return []

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().tolist()


# Singleton
image_embedder = ImageEmbedder()