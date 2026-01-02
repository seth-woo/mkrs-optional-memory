from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
from app.models.vision_llm import answer_image_question
from app.models.text_embedder import text_embedder
from app.models.image_embedder import image_embedder
from app.storage.qdrant_store import qdrant_store
from app.schemas.responses import SingleQAResponse
from app.core.logging import setup_logger

router = APIRouter()
logger = setup_logger("single_qa")


@router.post("/single", response_model=SingleQAResponse)
async def single_image_qa(
    images: list[UploadFile] = File(...),
    question: str = Form(...),
    save_to_memory: bool = Form(False),  # 🔑 Optional Memory toggle
):
    if not images:
        raise HTTPException(status_code=400, detail="No image uploaded")

    # ---------------------------
    # Load first image (Mode 1)
    # ---------------------------
    try:
        img = Image.open(images[0].file).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    logger.info(f"Received question: {question}")
    logger.info(f"Save to memory: {save_to_memory}")

    # ---------------------------
    # VLM inference
    # ---------------------------
    answer = answer_image_question(img, question)

    # ---------------------------
    # Optional persistence (Mode 1 memory)
    # ---------------------------
    if save_to_memory:
        try:
            image_embedding = image_embedder.embed(img)
            text_embedding = text_embedder.embed(f"Q: {question}\nA: {answer}")

            payload = {
                "question": question,
                "answer": answer,
                "filename": images[0].filename,
                "mode": "mode1",
            }

            qdrant_store.upsert_memory(
                text_embedding=text_embedding,
                image_embedding=image_embedding,
                payload=payload,
            )

            logger.info("Memory saved to Qdrant")

        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            # Memory is optional → do NOT fail request

    return SingleQAResponse(answer=answer)