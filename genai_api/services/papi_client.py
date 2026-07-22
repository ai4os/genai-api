"""
Client for interacting with the PAPI.
"""

import httpx
from fastapi import HTTPException

from genai_api.core.config import settings

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
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response

    async def get_or_create_genai_key(self) -> str:
        """
        Returns the user's LiteLLM API key for the GenAi environment.
        If it does not already exist in PAPI, a new one is created.
        """
        try: 
            response = await self._request("GET", "/v1/llm/api_keys")
            existing_keys = response.json()

            for key in existing_keys:
                if key.get("id") == GENAI_KEY_NAME:
                    return key["token"]

            create_response = await self._request(
                "POST", "/v1/llm/api_keys", params={"key_name": GENAI_KEY_NAME}
            )
            print('create response', create_response);
            token = create_response.json()

            return token
        except httpx.ReadTimeout:
            raise HTTPException(
                status_code=504,
                detail="Timeout calling PAPI",
            )