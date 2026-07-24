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

    async def stream_chat_completion(self, model, messages):
        yield "hola"
        yield " mundo"


app.dependency_overrides[get_user_litellm_client] = lambda: FakeLiteLLMClient()


@pytest.mark.asyncio
async def test_create_list_and_send_message():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create = await client.post(
            "/chats", json={"model": "claude-sonnet-5", "title": "Test"}
        )
        assert create.status_code == 201
        chat_id = create.json()["id"]

        listed = await client.get("/chats")
        assert listed.status_code == 200
        assert any(c["id"] == chat_id for c in listed.json())

        msg = await client.post(
            f"/chats/{chat_id}/messages",
            json={"model": "claude-sonnet-5", "content": "hola"},
        )

        assert msg.status_code == 200
        assert "event: done" in msg.text

        detail = await client.get(f"/chats/{chat_id}")
        assert detail.status_code == 200
        assert len(detail.json()["messages"]) == 2
