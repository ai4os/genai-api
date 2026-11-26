from datetime import datetime
from pydantic import BaseModel, Field


class UserCreditInfo(BaseModel):
    """
    Current team member budget and spend information.
    """

    spend: float = Field(
        description="Spend during the current team member budget cycle."
    )

    budget: float | None = Field(
        description="Maximum budget assigned to the team member."
    )

    remaining: float | None = Field(description="Remaining available budget.")

    usage_percentage: float | None = Field(
        description="Percentage of the budget consumed."
    )

    budget_reset_at: datetime | None = Field(
        default=None, description="Date and time when the current budget cycle resets."
    )
