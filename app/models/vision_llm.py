import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading VLM: {settings.MODEL_NAME}")

processor = AutoProcessor.from_pretrained(
    settings.MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    settings.MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model.eval()
logger.info("VLM loaded successfully")


def answer_image_question(image, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ],
        }
    ]

    # 🔑 THIS is the critical step
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    response = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    return response.strip()
