from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.router import router
from app.dependencies import set_llm
from app.services.llm_service import SmolLM


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize the LLM service
    llm_instance = SmolLM()
    set_llm(llm_instance)
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
