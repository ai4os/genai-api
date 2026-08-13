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
