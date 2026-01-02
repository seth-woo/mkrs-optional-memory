from fastapi import APIRouter
from app.storage.qdrant_store import qdrant_store
from app.core.logging import setup_logger

router = APIRouter()
logger = setup_logger("memory")


@router.get("/count")
def memory_count():
    """
    Return number of stored memory items in Qdrant.
    """
    count = qdrant_store.count()
    return {"count": count}


@router.post("/clear")
def clear_memory():
    """
    Clear all stored memory from Qdrant.
    """
    logger.warning("Clearing all MKRS memory")
    qdrant_store.clear()
    return {"status": "cleared"}