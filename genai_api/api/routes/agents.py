from fastapi import APIRouter, Depends

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
from genai_api.schemas.agent import AgentInfo, SkillInfo
from genai_api.services.litellm_client import LiteLLMClient

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentInfo])
async def list_available_models(
    user_id=Depends(get_current_user_id),
    litellm: LiteLLMClient = Depends(get_user_litellm_client),
):
    """
    Agents available in LiteLLM (using the user's API key).
    """
    agents = await litellm.list_agents()

    if not agents:
        return []

    return [
        AgentInfo(
            id=agent["agent_id"],
            name=agent["agent_name"],
            description=agent["agent_card_params"]["description"],
            url=agent["agent_card_params"]["url"],
            provider=agent["litellm_params"]["custom_llm_provider"],
            skills=[
                SkillInfo(
                    id=skill["id"],
                    name=skill["name"],
                    description=skill["description"],
                )
                for skill in agent["agent_card_params"]["skills"]
            ],
        )
        for agent in agents
    ]
