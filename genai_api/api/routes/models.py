from fastapi import APIRouter, Depends

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
from genai_api.schemas.model import ModelInfo
from genai_api.services.litellm_client import LiteLLMClient

router = APIRouter(prefix="/models", tags=["models"])

EXCLUDED_IDS = {"all-proxy-models"}
EXCLUDED_SUBSTRINGS = ("embedding",)


def _cost_per_million(cost_per_token) -> float | None:
    if cost_per_token is None:
        return None
    return float(cost_per_token) * 1_000_000


@router.get("", response_model=list[ModelInfo])
async def list_available_models(
    user_id=Depends(get_current_user_id),
    litellm: LiteLLMClient = Depends(get_user_litellm_client),
):
    """
    Models available in LiteLLM (using the user's API key).
    """
    user_models = await litellm.list_models()
    all_model_info = await litellm.get_model_info()
    status = await litellm.get_models_status()

    info_by_model = {m["model_name"]: m.get("model_info") or {} for m in all_model_info}
    status_by_model = {s["model_name"]: s["status"] for s in status.values()}

    # Remove manually models that include 'embedding'
    # and 'all-proxy-models' model from list
    visible_ids = [
        m["id"]
        for m in user_models
        if m["id"] not in EXCLUDED_IDS
        and not any(sub in m["id"].lower() for sub in EXCLUDED_SUBSTRINGS)
    ]

    return [
        ModelInfo(
            id=model_id,
            status=status_by_model.get(model_id, "unknown"),
            max_input_tokens=info_by_model.get(model_id, {}).get("max_input_tokens"),
            supports_vision=info_by_model.get(model_id, {}).get("supports_vision"),
            supports_reasoning=info_by_model.get(model_id, {}).get(
                "supports_reasoning"
            ),
            supports_function_calling=info_by_model.get(model_id, {}).get(
                "supports_function_calling"
            ),
            input_cost_per_million=_cost_per_million(
                info_by_model.get(model_id, {}).get("input_cost_per_token")
            ),
            output_cost_per_million=_cost_per_million(
                info_by_model.get(model_id, {}).get("output_cost_per_token")
            ),
        )
        for model_id in visible_ids
    ]
