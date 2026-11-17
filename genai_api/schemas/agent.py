from typing import List

from pydantic import BaseModel


class SkillInfo(BaseModel):
    id: str
    name: str
    description: str


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    url: str
    provider: str
    skills: List[SkillInfo]
    input_cost_per_million: float | None
    output_cost_per_million: float | None
    dev_status: str | None
    suggestions: List[str] = []
