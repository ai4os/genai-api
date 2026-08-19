import pytest
from httpx import ASGITransport, AsyncClient

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
from genai_api.main import app
from genai_api.services.litellm_client import LiteLLMClient
from tests.conftest import TEST_USER_ID


app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID


class FakeLiteLLMClient(LiteLLMClient):
    def __init__(self):
        pass

    async def list_models(self):
        return [
            {
                "id": "claude-sonnet-5",
                "model_info": {
                    "max_input_tokens": 200000,
                    "supports_vision": True,
                    "supports_reasoning": True,
                    "supports_function_calling": True,
                    "input_cost_per_token": 3e-6,
                    "output_cost_per_token": 1.5e-5,
                },
                "status": "healthy",
            },
            {
                "id": "gpt-4o",
                "model_info": {
                    "max_input_tokens": 128000,
                    "supports_vision": True,
                    "supports_reasoning": False,
                    "supports_function_calling": True,
                    "input_cost_per_token": 2.5e-6,
                    "output_cost_per_token": 1e-5,
                },
                "status": "unknown",
            },
        ]

    async def get_model_info(self):
        return [
            {
                "model_name": "claude-sonnet-5",
                "model_info": {
                    "max_input_tokens": 200000,
                    "supports_vision": True,
                    "supports_reasoning": True,
                    "supports_function_calling": True,
                    "input_cost_per_token": 3e-6,
                    "output_cost_per_token": 1.5e-5,
                },
            },
            {
                "model_name": "gpt-4o",
                "model_info": {
                    "max_input_tokens": 128000,
                    "supports_vision": True,
                    "supports_reasoning": False,
                    "supports_function_calling": True,
                    "input_cost_per_token": 2.5e-6,
                    "output_cost_per_token": 1e-5,
                },
            },
        ]

    async def get_models_status(self):
        return {
            "claude-sonnet-5": {
                "model_name": "claude-sonnet-5",
                "status": "healthy",
            },
            "gpt-4o": {
                "model_name": "gpt-4o",
                "status": "unknown",
            },
        }


app.dependency_overrides[get_user_litellm_client] = lambda: FakeLiteLLMClient()


@pytest.mark.asyncio
async def test_list_available_models():
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
    app.dependency_overrides[get_user_litellm_client] = lambda: FakeLiteLLMClient()

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/models")

        assert response.status_code == 200
        assert response.json() == [
            {
                "id": "claude-sonnet-5",
                "status": "healthy",
                "max_input_tokens": 200000,
                "supports_vision": True,
                "supports_reasoning": True,
                "supports_function_calling": True,
                "input_cost_per_million": 3.0,
                "output_cost_per_million": 15.0,
            },
            {
                "id": "gpt-4o",
                "status": "unknown",
                "max_input_tokens": 128000,
                "supports_vision": True,
                "supports_reasoning": False,
                "supports_function_calling": True,
                "input_cost_per_million": 2.5,
                "output_cost_per_million": 10.0,
            },
        ]
    finally:
        app.dependency_overrides.clear()
