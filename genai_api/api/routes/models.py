from fastapi import APIRouter, Depends

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
from genai_api.schemas.model import ModelInfo
from genai_api.services.litellm_client import LiteLLMClient

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def list_available_models(
    user_id=Depends(get_current_user_id),
    litellm: LiteLLMClient = Depends(get_user_litellm_client),
):
    """
    Models available in LiteLLM (using the user's API key).
    """
    models = await litellm.list_models()
    status = await litellm.get_models_status()
    print(f"Models status: {status}")

    # Add status to each model
    status_by_model = {s["model_name"]: s["status"] for s in status.values()}

    # Remove manually models that include'embedding'
    # and 'all-proxy-models' model from list
    models = [
        m
        for m in models
        if m["id"] != "all-proxy-models" and "embedding" not in m["id"].lower()
    ]

    return [
        ModelInfo(
            id=m["id"],
            status=status_by_model.get(m["id"], "unknown"),
        )
        for m in models
    ]
