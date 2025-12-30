from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.router import router
from app.services.llm_service import SmolLM
from app.dependencies import set_llm

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the LLM service
    llm_instance = SmolLM()
    set_llm(llm_instance)
    yield

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app

app = create_app()
