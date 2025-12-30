from app.services.llm_service import SmolLM

llm: SmolLM = None

def get_llm() -> SmolLM:
    return llm

def set_llm(instance: SmolLM):
    global llm
    llm = instance
