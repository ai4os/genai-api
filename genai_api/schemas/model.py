from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    status: str
    max_input_tokens: int | None = None
    supports_vision: bool | None = None
    supports_reasoning: bool | None = None
    supports_function_calling: bool | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
