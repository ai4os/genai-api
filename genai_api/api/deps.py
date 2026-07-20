import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from genai_api.core.security import TokenValidationError, decode_token
from genai_api.db.session import get_db
from genai_api.models.user import User

bearer_scheme = HTTPBearer()


class CurrentUser:
    def __init__(self, id: uuid.UUID, email: str | None, username: str | None, roles: list[str]):
        self.id = id
        self.email = email
        self.username = username
        self.roles = roles


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

    # Synchronizes the user inside the DB with the information in the token.
    # This means that 'users' always reflects the latest information Keycloak
    # has about the user, without requiring an separate registration step.
    user = await db.get(User, user_id)
    if user is None:
        user = User(id=user_id, email=email, username=username)
        db.add(user)
        await db.commit()
    elif user.email != email or user.username != username:
        user.email = email
        user.username = username
        await db.commit()

    return CurrentUser(id=user_id, email=email, username=username, roles=roles)


async def get_current_user_id(
    current_user: CurrentUser = Depends(get_current_user),
) -> uuid.UUID:
    return current_user.id


def require_role(role: str):
    """Dependency factory for protecting endpoints based on Keycloak roles.
    TODO: not used yet. Example of use: Depends(require_role("admin"))."""

    async def checker(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if role not in current_user.roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return current_user

    return checker
