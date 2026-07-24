import uuid

import pytest_asyncio

from genai_api.db.session import async_session, engine
from genai_api.models.user import User

TEST_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000099")


@pytest_asyncio.fixture(autouse=True)
async def ensure_test_user():
    # Creates a fake user for the tests.
    async with async_session() as session:
        user = await session.get(User, TEST_USER_ID)
        if user is None:
            session.add(
                User(id=TEST_USER_ID, email="test@example.com", username="test")
            )
        else:
            user.litellm_api_key = None
        await session.commit()
    yield

    await engine.dispose()
