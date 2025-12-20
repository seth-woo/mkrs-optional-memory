from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
from app.models.vision_llm import answer_image_question
from app.schemas.responses import SingleQAResponse
from app.core.logging import setup_logger

router = APIRouter()
logger = setup_logger("single_qa")

@router.post("/single", response_model=SingleQAResponse)
async def single_image_qa(
    images: list[UploadFile] = File(...),
    question: str = Form(...)
):
    if not images:
        raise HTTPException(status_code=400, detail="No image uploaded")

    try:
        img = Image.open(images[0].file).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    logger.info(f"Received question: {question}")

    answer = answer_image_question(img, question)

    return SingleQAResponse(answer=answer)
