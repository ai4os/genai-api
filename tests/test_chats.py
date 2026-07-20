import pytest
from httpx import ASGITransport, AsyncClient

from genai_api.api.deps import get_current_user_id
from genai_api.main import app
from tests.conftest import TEST_USER_ID

app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID


@pytest.mark.asyncio
async def test_create_list_and_send_message():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create = await client.post("/chats", json={"model": "claude-sonnet-5", "title": "Test"})
        assert create.status_code == 201
        chat_id = create.json()["id"]

        listed = await client.get("/chats")
        assert listed.status_code == 200
        assert any(c["id"] == chat_id for c in listed.json())

        msg = await client.post(f"/chats/{chat_id}/messages", json={"content": "hola"})
        assert msg.status_code == 201

        detail = await client.get(f"/chats/{chat_id}")
        assert detail.status_code == 200
        assert len(detail.json()["messages"]) == 1
