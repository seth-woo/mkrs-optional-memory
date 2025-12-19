import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from app.core.config import settings
from app.core.logging import setup_logger

logger = setup_logger("vision_llm")

logger.info(f"Loading VLM: {settings.MODEL_NAME}")

processor = AutoProcessor.from_pretrained(settings.MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(
    settings.MODEL_NAME,
    torch_dtype=torch.float16 if settings.DEVICE == "cuda" else torch.float32,
    device_map="auto" if settings.DEVICE == "cuda" else None
)

model.eval()
logger.info("VLM loaded successfully")


def answer_image_question(image, question: str) -> str:
    """
    Stateless single-image visual question answering
    """
    prompt = f"Answer the following question based on the image.\nQuestion: {question}"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    answer = processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return answer.strip()
