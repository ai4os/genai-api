"""
Client for interacting with the LiteLLM API.

The client uses the user's API key for all requests by default.

Endpoints that require administrative privileges (e.g. health/status)
explicitly use the service API key from settings.
"""

import json

import httpx
from fastapi import HTTPException

from a2a.client import A2ACardResolver

from genai_api.core.config import settings


class LiteLLMClient:
    def __init__(self, api_key: str):
        # User key is used for most of the requests. If needed, we use the service key from settings.
        self._api_key = api_key

    async def _request(
        self,
        method: str,
        path: str,
        *,
        api_key: str | None = None,
        **kwargs,
    ) -> httpx.Response:
        async with httpx.AsyncClient(base_url=settings.litellm_base_url) as client:
            response = await client.request(
                method,
                path,
                headers={"Authorization": f"Bearer {api_key or self._api_key}"},
                **kwargs,
            )

        if not response.is_success:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response

    ######## MODELS ########
    async def list_models(self) -> list[dict]:
        """Returns all the models configured in LiteLLM."""
        response = await self._request("GET", "/models")
        return response.json()["data"]

    async def get_models_status(self) -> list[dict]:
        # Users do not have access to this endpoint, so use the service API key.
        response = await self._request(
            "GET",
            "/health/latest",
            api_key=settings.litellm_api_key,
        )
        return response.json()["latest_health_checks"]

    async def chat_completion(self, model: str, messages: list[dict]) -> dict:
        """(OLD) Non-streaming version (kept in case it's needed somewhere)."""
        response = await self._request(
            "POST",
            "/chat/completions",
            json={"model": model, "messages": messages},
        )
        return response.json()

    async def stream_chat_completion(self, model_id: str, messages: list[dict]):
        """
        An asynchronous generator that produces chunks of text (content deltas) from the
        LiteLLM streaming endpoint (SSE, OpenAI-compatible format).
        """
        payload = {"model": model_id, "messages": messages, "stream": True}
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

    ######## AGENTS ########
    async def list_agents(self) -> list[dict]:
        """Returns all the agents configured in LiteLLM."""
        response = await self._request("GET", "/v1/agents")
        return response.json()

    async def stream_agent_run(self, agent_id: str, messages: list[dict]):
        import urllib.parse
        from uuid import uuid4

        parsed_url = urllib.parse.urlparse(settings.litellm_base_url)
        clean_base = f"{parsed_url.scheme}://{parsed_url.netloc}"
        base_url = f"{clean_base}/a2a/{agent_id}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        last_content = messages[-1]["content"] if messages else ""

        async with httpx.AsyncClient(headers=headers, timeout=None) as http_client:
            resolver = A2ACardResolver(httpx_client=http_client, base_url=base_url)
            try:
                agent_card = await resolver.get_agent_card()
                # Extract the URL from the first interface and force HTTPS.
                # Otherwise, the server sends the request over HTTP, since it is
                # treated as a local request between machines on the same network.
                target_url = agent_card.supported_interfaces[0].url
                if target_url.startswith("http://"):
                    target_url = target_url.replace("http://", "https://")
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error al resolver Agent Card: {str(e)}"
                )

            # LiteLLM payload (message/stream)
            request_id = str(uuid4())
            message_id = uuid4().hex

            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "message/send",  # TODO: change to message/stream
                "params": {
                    "message": {
                        "kind": "message",
                        "messageId": message_id,
                        "role": "user",
                        "parts": [{"kind": "text", "text": last_content}],
                    }
                },
            }

            try:
                async with http_client.stream(
                    "POST", target_url, json=payload
                ) as response:
                    if not response.is_success:
                        body = await response.aread()
                        raise HTTPException(
                            status_code=response.status_code, detail=body.decode()
                        )

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue

                        if line == "data: [DONE]" or line == "[DONE]":
                            break

                        if line.startswith("data:"):
                            line = line[len("data:") :].strip()

                        try:
                            chunk = json.loads(line)
                            result = chunk.get("result", {})

                            if (
                                result.get("kind") == "artifact-update"
                                and "artifact" in result
                            ):
                                for part in result["artifact"].get("parts", []):
                                    if part.get("kind") == "text":
                                        content = part.get("text")
                                        if content:
                                            yield content

                            elif "message" in result:
                                for part in result["message"].get("parts", []):
                                    if part.get("kind") == "text":
                                        content = part.get("text")
                                        if content:
                                            yield content

                            elif (
                                result.get("kind") == "status-update"
                                and result.get("final") is True
                            ):
                                break

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error durante la conexión HTTP: {str(e)}"
                )
