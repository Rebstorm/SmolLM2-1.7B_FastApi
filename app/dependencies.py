from typing import Optional

from app.services.llm_service import SmolLM
from app.services.rag_service import CVRagService

llm: Optional[SmolLM] = None
rag: Optional[CVRagService] = None


def get_llm() -> Optional[SmolLM]:
    return llm


def set_llm(instance: SmolLM) -> None:
    global llm
    llm = instance


def get_rag() -> Optional[CVRagService]:
    return rag


def set_rag(instance: CVRagService) -> None:
    global rag
    rag = instance
