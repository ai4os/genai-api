"""
Client for interacting with the PAPI.
"""

import logging

import httpx
from fastapi import HTTPException

from genai_api.core.config import settings

logger = logging.getLogger(__name__)

GENAI_KEY_NAME = "genai_app"


class PAPIClient:
    def __init__(self, token: str):
        self._token = token

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(base_url=settings.papi_base_url) as client:
            response = await client.request(
                method,
                path,
                headers={"Authorization": f"Bearer {self._token}"},
                **kwargs,
            )
        if not response.is_success:
            logger.warning("PAPI %s %s answered %s", method, path, response.status_code)
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response

    async def create_genai_key(self) -> str:
        """
        Creates a fresh LiteLLM API key for the GenAI environment.

        The method first checks whether an API key with the configured
        ``GENAI_KEY_NAME`` already exists in PAPI. If it does, the key is deleted
        before creating a new one.

        This guarantees that the key stored in PAPI and the one persisted in the
        GenAI database are always synchronized. An existing key may indicate that
        it was created previously but was never stored (or is no longer stored) in
        the GenAI database.

        Returns:
            The newly created API key token.
        """
        try:
            response = await self._request("GET", "/v1/llm/api_keys")

            for key in response.json():
                if key.get("id") == GENAI_KEY_NAME:
                    logger.warning(
                        "A '%s' key already existed in PAPI but not in our database; "
                        "replacing it",
                        GENAI_KEY_NAME,
                    )
                    await self._request(
                        "DELETE",
                        "/v1/llm/api_keys",
                        params={"key_name": GENAI_KEY_NAME},
                    )
                    break

            create_response = await self._request(
                "POST", "/v1/llm/api_keys", params={"key_name": GENAI_KEY_NAME}
            )
            # Log only the status: the body is the secret key itself.
            logger.info(
                "Created '%s' key in PAPI (%s)",
                GENAI_KEY_NAME,
                create_response.status_code,
            )
            return create_response.json()
        except httpx.TimeoutException as exc:
            logger.error("Timeout calling PAPI")
            raise HTTPException(status_code=504, detail="Timeout calling PAPI") from exc
        except httpx.HTTPError as exc:
            logger.exception("Could not reach PAPI")
            raise HTTPException(status_code=502, detail="Could not reach PAPI") from exc
