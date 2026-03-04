from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import re
try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover - fallback when langchain is not installed
    ChatPromptTemplate = None

from app.models.vision_llm import answer_image_question
from app.models.text_embedder import text_embedder
from app.models.image_embedder import image_embedder
from app.storage.qdrant_store import qdrant_store
from app.schemas.responses import SingleQAResponse
from app.core.config import settings
from app.core.logging import setup_logger

router = APIRouter()
logger = setup_logger("multi_qa")

COLOR_TERMS = [
    "red", "green", "yellow", "orange", "blue", "purple", "pink",
    "brown", "black", "white", "gold", "silver",
]

FRUIT_TERMS = [
    "apple", "banana", "orange", "raspberry", "raspberries", "strawberry",
    "strawberries", "grape", "grapes", "pear", "peach", "mango", "pineapple",
    "watermelon", "kiwi", "lemon", "lime", "cherry", "cherries", "blueberry",
    "blueberries",
]


RAG_PROMPT = None
if ChatPromptTemplate is not None:
    RAG_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a vision-language assistant with external memory. "
                    "Collection memory records are factual and should be used when the user asks about the collection/history."
                ),
            ),
            (
                "human",
                (
                    "User question:\n{question}\n\n"
                    "Collection memory records:\n{context}\n\n"
                    "Instructions:\n"
                    "1) Identify the fruit and color in the current image.\n"
                    "2) If the user asks about another fruit in the collection, compare against ALL listed memory records.\n"
                    "3) Name matching fruits explicitly and include the matching color.\n"
                    "4) If no relevant match exists in memory, say so clearly.\n"
                    "5) Do not answer only about the current image when the question asks about the collection."
                ),
            ),
        ]
    )


def _build_memory_context(memory_hits) -> str:
    if not memory_hits:
        return "No memory retrieved."

    lines = []
    for i, hit in enumerate(memory_hits, start=1):
        payload = hit.payload or {}
        filename = payload.get("filename", "unknown")
        q = payload.get("question", "")
        a = payload.get("answer", "")
        lines.append(f"[{i}] file={filename} | question={q} | answer={a}")
    return "\n".join(lines)


def _build_augmented_question(question: str, memory_context: str) -> str:
    if RAG_PROMPT is not None:
        prompt_value = RAG_PROMPT.invoke(
            {
                "question": question,
                "context": memory_context,
            }
        )
        return prompt_value.messages[-1].content

    logger.warning("LangChain is unavailable. Falling back to plain prompt formatting.")
    return (
        f"User question:\n{question}\n\n"
        f"Collection memory records:\n{memory_context}\n\n"
        "Instructions:\n"
        "1) Identify the fruit and color in the current image.\n"
        "2) If the user asks about another fruit in the collection, compare against ALL listed memory records.\n"
        "3) Name matching fruits explicitly and include the matching color.\n"
        "4) If no relevant match exists in memory, say so clearly.\n"
        "5) Do not answer only about the current image when the question asks about the collection."
    )


def _extract_term(text: str, terms: list[str]) -> str | None:
    lowered = (text or "").lower()
    for term in sorted(terms, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return term
    return None


def _normalize_fruit_name(fruit: str | None) -> str | None:
    if fruit is None:
        return None
    aliases = {
        "raspberries": "raspberry",
        "strawberries": "strawberry",
        "grapes": "grape",
        "cherries": "cherry",
        "blueberries": "blueberry",
    }
    return aliases.get(fruit, fruit)


def _extract_memory_fruit_color(payload: dict) -> tuple[str | None, str | None]:
    answer = payload.get("answer", "")
    question = payload.get("question", "")
    combined = f"{answer}\n{question}"
    fruit = _normalize_fruit_name(_extract_term(combined, FRUIT_TERMS))
    color = _extract_term(combined, COLOR_TERMS)
    return fruit, color


def _needs_collection_color_compare(question: str) -> bool:
    q = question.lower()
    return (
        "collection" in q
        and ("same color" in q or "shares the same color" in q or "another fruit" in q)
    )


def _enforce_color_match_answer(question: str, vlm_answer: str, memory_hits) -> str:
    if not _needs_collection_color_compare(question):
        return vlm_answer

    current_fruit = _normalize_fruit_name(_extract_term(vlm_answer, FRUIT_TERMS))
    current_color = _extract_term(vlm_answer, COLOR_TERMS)

    if not current_fruit or not current_color:
        logger.info("Could not extract current fruit/color from VLM answer; keeping original response.")
        return vlm_answer

    matching_fruits = []
    for hit in memory_hits:
        payload = hit.payload or {}
        mem_fruit, mem_color = _extract_memory_fruit_color(payload)
        if mem_fruit and mem_color == current_color and mem_fruit != current_fruit:
            matching_fruits.append(mem_fruit)

    matching_fruits = sorted(set(matching_fruits))
    if matching_fruits:
        joined = ", ".join(matching_fruits)
        return (
            f"The fruit on the book is {current_fruit}, which is {current_color}. "
            f"Another fruit in the collection that shares the same color is {joined}."
        )

    return (
        f"The fruit on the book is {current_fruit}, which is {current_color}. "
        "There is no other fruit in the collection that shares the same color."
    )


@router.post("/multi", response_model=SingleQAResponse)
async def multi_image_qa(
    images: list[UploadFile] = File(...),
    question: str = Form(...),
    save_to_memory: bool = Form(False),
):
    logger.info("Entered /qa/multi endpoint")
    if not images:
        raise HTTPException(status_code=400, detail="No image uploaded")

    try:
        img = Image.open(images[0].file).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    logger.info(f"Received question: {question}")
    logger.info(f"Save to Memory: {save_to_memory}")

    memory_count = qdrant_store.count()
    if memory_count == 0:
        logger.info("No memory found in Qdrant. Falling back to plain VLM response.")
        answer = answer_image_question(img, question)
        return SingleQAResponse(answer=answer)

    try:
        text_query_embedding = text_embedder.embed([question])[0]
        image_query_embedding = image_embedder.embed([img])[0]

        # For small collections, include all records to avoid missing cross-item comparisons.
        if memory_count <= 25:
            hits = qdrant_store.list_memories(limit=memory_count)
        else:
            hits = qdrant_store.search_hybrid(
                text_query_embedding=text_query_embedding,
                image_query_embedding=image_query_embedding,
                top_k=settings.RAG_TOP_K,
            )

        memory_context = _build_memory_context(hits)
        logger.info(f"multi_qa using {len(hits)} memory record(s)")
        augmented_question = _build_augmented_question(question, memory_context)
        answer = answer_image_question(img, augmented_question)
        answer = _enforce_color_match_answer(question, answer, hits)

        if save_to_memory:
            text_embedding = text_embedder.embed([f"Q: {question}\nA: {answer}"])[0]
            image_embedding = image_embedder.embed([img])[0]
            payload = {
                "question": question,
                "answer": answer,
                "filename": images[0].filename,
                "mode": "mode2",
            }
            qdrant_store.upsert_memory(
                text_embedding=text_embedding,
                image_embedding=image_embedding,
                payload=payload,
            )
            logger.info("Memory saved to Qdrant")

        return SingleQAResponse(answer=answer)

    except Exception as e:
        logger.error(f"multi_qa failed, falling back to direct answer: {e}")
        answer = answer_image_question(img, question)
        return SingleQAResponse(answer=answer)
