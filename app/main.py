import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.router import router
from app.dependencies import set_llm, set_rag
from app.services.llm_service import SmolLM
from app.services.rag_service import CVRagService

CV_PATH = os.getenv("CV_PATH", "data/cv.pdf")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize the LLM service
    llm_instance = SmolLM()
    set_llm(llm_instance)

    # Initialize the RAG service and index the CV if present
    rag_instance = CVRagService(persist_dir=CHROMA_PERSIST_DIR)
    if os.path.exists(CV_PATH):
        rag_instance.index_cv(CV_PATH)
    else:
        print(f"Warning: CV_PATH '{CV_PATH}' not found; RAG will return no context.")
    set_rag(rag_instance)

    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
