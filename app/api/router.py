from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.dependencies import get_llm, get_rag
from app.services.llm_service import SmolLM
from app.services.rag_service import CVRagService

router = APIRouter()


class ConfigUpdate(BaseModel):
    system_prompt: Optional[str] = None
    max_new_tokens: Optional[int] = None


@router.get("/generate")
async def generate(
    query: str = Query(..., description="The prompt for the model"),
    stream: bool = Query(False, description="Whether to stream the response"),
    max_new_tokens: Optional[int] = Query(
        None, description="Maximum new tokens to generate"
    ),
    use_rag: bool = Query(
        False, description="Whether to retrieve CV context via RAG"
    ),
    top_k: int = Query(
        3, description="Number of CV chunks to retrieve when use_rag=true"
    ),
    llm: SmolLM = Depends(get_llm),
    rag: Optional[CVRagService] = Depends(get_rag),
) -> Any:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    context: Optional[str] = None
    if use_rag:
        if rag is None:
            raise HTTPException(status_code=503, detail="RAG service not loaded")
        chunks = rag.query(query, top_k=top_k)
        context = "\n\n".join(chunks) if chunks else None

    if stream:
        return StreamingResponse(
            llm.stream_generate(
                query, max_new_tokens=max_new_tokens, context=context
            ),
            media_type="text/plain",
        )

    result = llm.generate(query, max_new_tokens=max_new_tokens, context=context)
    response: Dict[str, Any] = {
        "prompt": query,
        "response": result,
        "used_rag": use_rag,
    }
    if context:
        response["retrieved_chunks"] = len(context.split("\n\n"))
    return response


@router.post("/config")
async def update_config(
    config: ConfigUpdate, llm: SmolLM = Depends(get_llm)
) -> Dict[str, Any]:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    llm.update_config(
        system_prompt=config.system_prompt, max_new_tokens=config.max_new_tokens
    )
    return {
        "status": "success",
        "config": {
            "system_prompt": llm.set_default_system_prompt,
            "max_new_tokens": llm.max_new_tokens,
        },
    }
