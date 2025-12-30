from typing import Optional

from app.services.llm_service import SmolLM

llm: Optional[SmolLM] = None


def get_llm() -> Optional[SmolLM]:
    return llm


def set_llm(instance: SmolLM) -> None:
    global llm
    llm = instance
