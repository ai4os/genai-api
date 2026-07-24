"""
Client for interacting with the LiteLLM API.

By default, it uses the service API key (the generic one defined in the
settings), intended for operations that are not tied to a specific user.

For operations that need to be performed on behalf of a specific
user (e.g., tracking usage or costs per user), the client can be
instantiated with that user's API key: LiteLLMClient(api_key=user_key)
"""

import json

import httpx
from fastapi import HTTPException

from genai_api.core.config import settings


class LiteLLMClient:
    def __init__(self, api_key: str | None = None):
        # If no user  key is provided, the service key is used
        self._api_key = api_key or settings.litellm_api_key

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(base_url=settings.litellm_base_url) as client:
            response = await client.request(
                method,
                path,
                headers={"Authorization": f"Bearer {self._api_key}"},
                **kwargs,
            )
        if not response.is_success:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response

    async def list_models(self) -> list[dict]:
        """Returns all the models configured in LiteLLM."""
        response = await self._request("GET", "/models")
        return response.json()["data"]

    async def chat_completion(self, model: str, messages: list[dict]) -> dict:
        """(OLD) Non-streaming version (kept in case it's needed somewhere)."""
        response = await self._request(
            "POST",
            "/chat/completions",
            json={"model": model, "messages": messages},
        )
        return response.json()

    async def stream_chat_completion(self, model: str, messages: list[dict]):
        """
        An asynchronous generator that produces chunks of text (content deltas) from the
        LiteLLM streaming endpoint (SSE, OpenAI-compatible format).
        """
        payload = {"model": model, "messages": messages, "stream": True}
        # timeout=None: a streaming response may take quite a long time to
        # complete; httpx’s default timeout would cut it short prematurely.
        async with httpx.AsyncClient(
            base_url=settings.litellm_base_url, timeout=None
        ) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            ) as response:
                if not response.is_success:
                    body = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code, detail=body.decode()
                    )

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content")
                    if content:
                        yield content
