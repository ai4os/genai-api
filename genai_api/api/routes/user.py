from fastapi import APIRouter, Depends, HTTPException

from genai_api.schemas.user import UserCreditInfo
from genai_api.services.litellm_client import LiteLLMClient
from genai_api.api.deps import get_current_user_id, get_user_litellm_client


router = APIRouter(prefix="/user", tags=["user"])


@router.get("/credit_info")
async def get_credit_info(
    user_id: str = Depends(get_current_user_id),
    litellm: LiteLLMClient = Depends(get_user_litellm_client),
) -> UserCreditInfo:
    """
    Retrieves the authenticated user's current Team Member budget.
    Returns:
        UserCreditInfo with spend, budget, remaining balance,
        usage percentage and budget reset date.
    """

    # Retrieve user information
    user_data = await litellm.get_user_info(user_id)

    # Get the user's team memberships
    teams = user_data.get("teams", [])

    if not teams:
        raise HTTPException(
            status_code=404, detail="User does not belong to any LiteLLM team."
        )

    team = teams[0]
    team_id = team.get("team_id")

    # Retrieve the team info
    team_data = await litellm.get_team_info(team_id)

    team_info = team_data.get("team_info", {})

    team_budget_table = team_info.get("team_member_budget_table") or {}

    # Get the user's budget
    budget = team_budget_table.get("max_budget")

    # Get the user's spent budget
    team_memberships = team_data.get("team_memberships", [])
    user_id_str = str(user_id)
    team_membership = next(
        (item for item in team_memberships if item.get("user_id") == user_id_str), None
    )

    if team_membership:
        spend = team_membership.get("spend")

    spend = float(spend or 0)

    # Calculate available balance
    if budget is not None:
        budget = float(budget)
        remaining = max(0.0, budget - spend)

        usage_percentage = (spend / budget) * 100 if budget > 0 else None
    else:
        remaining = None
        usage_percentage = None

    # Get the budget reset date
    reset_at = (
        team_budget_table.get("budget_reset_at")
        if budget is not None and "team_budget_table" in locals()
        else None
    )

    return UserCreditInfo(
        spend=round(spend, 6),
        budget=budget,
        remaining=round(remaining, 6) if remaining is not None else None,
        usage_percentage=round(usage_percentage, 2)
        if usage_percentage is not None
        else None,
        budget_reset_at=reset_at,
    )
