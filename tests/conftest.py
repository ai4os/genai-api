"""
Shared test setup.

- Tests run against the real `genai_test` Postgres database (migrated with Alembic),
  never against `genai_dev`:  (set -a; source .env.test; set +a; pytest)
- Authentication and LiteLLM are replaced through FastAPI dependency overrides,
  so no Keycloak, PAPI or LiteLLM is needed.
- Milvus is never contacted: tests that go through it monkeypatch `services.milvus`.
"""

import uuid

import pytest_asyncio
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
from genai_api.db.session import async_session, engine
from genai_api.main import app
from genai_api.models.user import User
from genai_api.services.litellm_client import LiteLLMClient

TEST_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000099")
OTHER_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000098")


class FakeLiteLLMClient(LiteLLMClient):
    """Configurable stand-in. Set the attributes in each test to shape the reply."""

    def __init__(self):
        super().__init__(api_key="fake")
        self.chunks: list[str] = ["hola", " mundo"]
        self.error: HTTPException | None = None  # raised after the chunks
        self.capabilities: dict = {}
        # What the code under test sent us
        self.last_messages: list[dict] = []
        self.last_collection_ids: list[str] | None = None

    async def get_model_capabilities(self, model_id: str) -> dict:
        return self.capabilities

    async def _reply(self):
        for text in self.chunks:
            yield "content", text
        if self.error:
            raise self.error

    async def stream_chat_completion(self, model_id, messages, collection_ids=None):
        self.last_messages = messages
        self.last_collection_ids = collection_ids
        async for item in self._reply():
            yield item

    async def stream_agent_run(self, chat_id, agent_id, messages, collection_ids=None):
        self.last_messages = messages
        async for item in self._reply():
            yield item


@pytest_asyncio.fixture(autouse=True)
async def ensure_test_users():
    async with async_session() as session:
        for user_id, name in ((TEST_USER_ID, "test"), (OTHER_USER_ID, "other")):
            await session.merge(
                User(id=user_id, email=f"{name}@example.com", username=name)
            )
        await session.commit()
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def fake_llm() -> FakeLiteLLMClient:
    return FakeLiteLLMClient()


@pytest_asyncio.fixture
async def client(fake_llm):
    """HTTP client acting as TEST_USER, with LiteLLM faked."""
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID
    app.dependency_overrides[get_user_litellm_client] = lambda: fake_llm
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
