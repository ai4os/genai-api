from fastapi import APIRouter, Depends

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
from genai_api.schemas.agent import AgentInfo, SkillInfo
from genai_api.services.litellm_client import LiteLLMClient

router = APIRouter(prefix="/agents", tags=["agents"])


def _cost_per_million(cost_per_token) -> float | None:
    return None if cost_per_token is None else float(cost_per_token) * 1_000_000


def _to_agent_info(agent: dict) -> AgentInfo:
    """Maps a LiteLLM agent record to AgentInfo. Optional fields use .get() so an
    agent without skills, costs or metadata does not break the whole listing."""
    card = agent.get("agent_card_params") or {}
    params = agent.get("litellm_params") or {}
    skills = card.get("skills") or []

    return AgentInfo(
        id=agent["agent_id"],
        name=agent["agent_name"],
        description=card.get("description", ""),
        url=card.get("url", ""),
        # `custom_llm_provider` no longer exists in LiteLLM's agent params.
        # TODO: decide where the provider should come from (the UI uses it for the icon).
        provider=params.get("custom_llm_provider") or "unknown",
        skills=[
            SkillInfo(id=s["id"], name=s["name"], description=s.get("description", ""))
            for s in skills
        ],
        input_cost_per_million=_cost_per_million(params.get("input_cost_per_token")),
        output_cost_per_million=_cost_per_million(params.get("output_cost_per_token")),
        dev_status=(params.get("metadata") or {}).get("dev_status"),
        suggestions=(skills[0].get("examples") or []) if skills else [],
    )


@router.get("", response_model=list[AgentInfo])
async def list_agents(
    user_id=Depends(get_current_user_id),
    litellm: LiteLLMClient = Depends(get_user_litellm_client),
):
    """Agents available in LiteLLM (with the user's own key)."""
    agents = await litellm.list_agents()
    return [_to_agent_info(a) for a in agents or []]
