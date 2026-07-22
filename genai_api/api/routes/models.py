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
    Models available in LiteLLM (con la API key propia del usuario).
    """
    models = await litellm.list_models()
    # TODO: remove manually 'all-proxy-models' from list until we discover why
    # it shows in the list
    models = [m for m in models if m["id"] != "all-proxy-models"]
    
    return [ModelInfo(id=m["id"]) for m in models]