"""
FastAPI dependencies: authentication, current user and per-user LiteLLM client.

Dependency chain (FastAPI resolves it per request, and every dependency that
asks for `get_db` receives the SAME session within that request):

    bearer token ──► get_current_user ──► get_current_user_id
                          │
                          └──► get_user_litellm_client
"""

import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from genai_api.core.security import TokenValidationError, decode_token
from genai_api.db.session import async_session, get_db
from genai_api.models.user import User
from genai_api.services.litellm_client import LiteLLMClient
from genai_api.services.papi_client import PAPIClient

bearer_scheme = HTTPBearer()


@dataclass(frozen=True)
class CurrentUser:
    id: uuid.UUID
    email: str | None
    username: str | None
    roles: list[str]
    # Stored only to forward it to downstream services (PAPI)
    # within the same request. It is not persisted anywhere.
    raw_token: str


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    try:
        payload = decode_token(credentials.credentials)
    except TokenValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = uuid.UUID(payload["sub"])
    email = payload.get("email")
    username = payload.get("preferred_username")
    roles = payload.get("realm_access", {}).get("roles", [])

    # One atomic statement: creates the user, or updates email/username only if
    # they changed. Being atomic, it also avoids the race where parallel requests
    # of a brand-new user (/chats, /agents, /models fired together by the frontend).
    stmt = pg_insert(User).values(id=user_id, email=email, username=username)
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={"email": email, "username": username},
        where=User.email.is_distinct_from(email)
        | User.username.is_distinct_from(username),
    )
    await db.execute(stmt)
    await db.commit()

    return CurrentUser(
        id=user_id,
        email=email,
        username=username,
        roles=roles,
        raw_token=credentials.credentials,
    )


async def get_current_user_id(
    current_user: CurrentUser = Depends(get_current_user),
) -> uuid.UUID:
    return current_user.id


def require_role(role: str):
    """Dependency factory to protect endpoints by Keycloak realm role.
    TODO: Not used yet. Example: Depends(require_role("admin"))."""

    async def checker(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role"
            )
        return current_user

    return checker


async def refresh_user_litellm_key(
    user_id: uuid.UUID, raw_token: str, failed_key: str
) -> str:
    """
    Regenerates the user's LiteLLM key after LiteLLM rejected `failed_key`.
    """
    async with async_session() as session:
        user = (
            await session.execute(
                select(User).where(User.id == user_id).with_for_update()
            )
        ).scalar_one()
        if user.litellm_api_key == failed_key:
            user.litellm_api_key = await PAPIClient(token=raw_token).create_genai_key()
        new_key = user.litellm_api_key
        await session.commit()
    return new_key


async def get_user_litellm_client(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LiteLLMClient:
    """
    LiteLLMClient configured with the user's own LiteLLM key.

    The key is created through PAPI (with the user's token) the first time it is
    needed, and cached in `users.litellm_api_key`.

    Concurrency: for a brand-new user the frontend fires several requests in
    parallel and all of them would see "no key yet". Only in that case we lock
    the user's row (`SELECT ... FOR UPDATE`), so the first request creates the key
    and the rest wait, then find it already stored. Users that already have a key
    never take the lock, so their parallel requests do not queue.

    Safety net: if LiteLLM later says it does not know the cached key
    (`token_not_found_in_db`), the client calls `refresh_user_litellm_key`
    and retries once.
    """
    user = await db.get(User, current_user.id)

    if user.litellm_api_key is None:
        user = (
            await db.execute(
                select(User)
                .where(User.id == current_user.id)
                .with_for_update()
                .execution_options(populate_existing=True)  # refresh after waiting
            )
        ).scalar_one()
        if user.litellm_api_key is None:  # re-check now that we hold the lock
            user.litellm_api_key = await PAPIClient(
                token=current_user.raw_token
            ).create_genai_key()
        await db.commit()  # also releases the lock

    async def on_invalid_key(failed_key: str) -> str:
        return await refresh_user_litellm_key(
            current_user.id, current_user.raw_token, failed_key
        )

    return LiteLLMClient(
        api_key=user.litellm_api_key,
        user_token=current_user.raw_token,
        on_invalid_key=on_invalid_key,
    )
