from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from genai_api.api.deps import get_current_user_id
from genai_api.main import app
from tests.conftest import TEST_USER_ID

app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID


@pytest.mark.asyncio
async def test_list_available_models():
    fake_models = [{"id": "claude-sonnet-5"}, {"id": "gpt-4o"}]

    with patch(
        "genai_api.api.routes.models.LiteLLMClient.list_models",
        new=AsyncMock(return_value=fake_models),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/models")

    assert response.status_code == 200
    assert response.json() == [{"id": "claude-sonnet-5"}, {"id": "gpt-4o"}]
