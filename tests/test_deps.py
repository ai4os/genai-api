from unittest.mock import AsyncMock, patch

import pytest

from genai_api.api.deps import get_user_litellm_client, CurrentUser
from genai_api.db.session import async_session
from tests.conftest import TEST_USER_ID


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
