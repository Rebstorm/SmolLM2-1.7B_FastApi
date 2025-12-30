from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.dependencies import get_llm
from app.services.llm_service import SmolLM

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
    llm: SmolLM = Depends(get_llm),
) -> Any:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if stream:
        return StreamingResponse(
            llm.stream_generate(query, max_new_tokens=max_new_tokens),
            media_type="text/plain",
        )

    result = llm.generate(query, max_new_tokens=max_new_tokens)
    return {"prompt": query, "response": result}


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
