from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.single_qa import router as single_qa_router
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(single_qa_router, prefix="/qa", tags=["Mode 1 - Single Image QA"])

@app.get("/health")
def health_check():
    return {"status": "MKRS Mode 1 is running"}

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")