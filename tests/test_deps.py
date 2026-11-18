from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from genai_api.api.deps import (
    CurrentUser,
    get_user_litellm_client,
    refresh_user_litellm_key,
)
from genai_api.db.session import async_session
from genai_api.models.user import User
from genai_api.services.papi_client import PAPIClient
from tests.conftest import TEST_USER_ID


@pytest_asyncio.fixture(autouse=True)
async def reset_user_litellm_key():
    async def _reset():
        async with async_session() as session:
            user = await session.get(User, TEST_USER_ID)
            user.litellm_api_key = None
            await session.commit()

    await _reset()
    yield
    await _reset()


@pytest.mark.asyncio
async def test_get_user_litellm_client_creates_key_if_missing():
    fake_user = CurrentUser(
        id=TEST_USER_ID,
        email="test@example.com",
        username="test",
        roles=[],
        raw_token="fake-token",
    )

    with patch(
        "genai_api.api.deps.PAPIClient.create_genai_key",
        new=AsyncMock(return_value="sk-nueva-key"),
    ) as mock_papi:
        async with async_session() as db:
            client = await get_user_litellm_client(current_user=fake_user, db=db)

        assert client._api_key == "sk-nueva-key"
        mock_papi.assert_called_once()

        async with async_session() as db:
            await get_user_litellm_client(current_user=fake_user, db=db)
        assert mock_papi.call_count == 1


@pytest.mark.asyncio
async def test_refresh_only_regenerates_if_key_is_the_failed_one(monkeypatch):
    created = []

    async def fake_create(self) -> str:
        created.append(1)
        return "new-key"

    monkeypatch.setattr(PAPIClient, "create_genai_key", fake_create)

    async with async_session() as session:
        user = await session.get(User, TEST_USER_ID)
        user.litellm_api_key = "old-key"
        await session.commit()

    assert await refresh_user_litellm_key(TEST_USER_ID, "tok", "old-key") == "new-key"
    # A parallel request that failed with the same old key must NOT regenerate again
    assert await refresh_user_litellm_key(TEST_USER_ID, "tok", "old-key") == "new-key"
    assert len(created) == 1
