from functools import lru_cache

from fastapi import HTTPException
import jwt
from jwt import PyJWKClient

from genai_api.core.config import settings


class TokenValidationError(Exception):
    """The token is invalid, has expired or fails the key/issuer/audience verification."""


@lru_cache
def get_jwk_client() -> PyJWKClient:
    # PyJWKClient downloads and caches Keycloak's public keys (JWKS),
    # and automatically refreshes them if it encounters a key ID (KID)
    # it does not yet recognize (e.g., after a key rotation in Keycloak).
    jwks_url = f"{settings.keycloak_issuer}/protocol/openid-connect/certs"
    return PyJWKClient(jwks_url)


def decode_token(token: str) -> dict:
    jwk_client = get_jwk_client()
    try:
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.keycloak_issuer,
            options={"verify_aud": False}
        )
    except Exception as e:
        print(f"!!! JWT ERROR: {type(e).__name__} - {str(e)}") 
        raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError as exc:
        raise TokenValidationError(str(exc)) from exc
    return payload
