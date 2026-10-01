"""
Client for interacting with the LiteLLM API.

The client uses the user's API key for all requests by default.

Endpoints that require administrative privileges (e.g. health/status)
explicitly use the service API key from settings.
"""

import json

import httpx
from fastapi import HTTPException

from asyncache import cached
from cachetools import TTLCache
from cachetools.keys import hashkey

from a2a.client import A2ACardResolver

from genai_api.core.config import settings

# Alias of the Milvus MCP server registered in LiteLLM and its tools
MILVUS_MCP_SERVER = "knowledge_milvus"
MILVUS_MCP_TOOL = "search"


class LiteLLMClient:
    def __init__(self, api_key: str = "", user_token: str | None = None):
        # User key is used for most of the requests. If needed, we use the service key from settings.
        self._api_key = api_key
        # User token is only sent inside the Milvus MCP tool definition (see
        # `_milvus_mcp_tool`), so that it is never exposed to other tools/MCPs.
        self._user_token = user_token

    def _get_headers(
        self, api_key: str | None = None, extra_headers: dict | None = None
    ) -> dict:
        headers = {"Authorization": f"Bearer {api_key or self._api_key}"}
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _milvus_mcp_tool(self) -> dict:
        """
        MCP tool definition for LiteLLM's /chat/completions. With
        `require_approval: "never"` LiteLLM runs the tool calls itself and streams
        back the final answer, so we do not need to handle tool calls here.
        """
        tool = {
            "type": "mcp",
            "server_label": MILVUS_MCP_SERVER,
            "server_url": f"litellm_proxy/mcp/{MILVUS_MCP_SERVER}",
            "allowed_tools": [MILVUS_MCP_TOOL],
            "require_approval": "never",
        }
        if self._user_token:
            tool["headers"] = {"x-user-token": self._user_token}
        return tool

    @staticmethod
    def _collections_system_message(collection_ids: list[str]) -> dict:
        return {
            "role": "system",
            "content": (
                "You can search the user's knowledge base with the "
                f"`{MILVUS_MCP_TOOL}` tool. When the question may be answered by "
                "the documents, call it with "
                f"collections={json.dumps(collection_ids)} (use exactly these ids). "
                "Base your answer on the retrieved passages and mention the source "
                "filename. If nothing relevant is found, say so instead of guessing."
            ),
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        api_key: str | None = None,
        headers: dict | None = None,
        **kwargs,
    ) -> httpx.Response:
        req_headers = self._get_headers(api_key=api_key, extra_headers=headers)
        async with httpx.AsyncClient(base_url=settings.litellm_base_url) as client:
            response = await client.request(
                method,
                path,
                headers=req_headers,
                **kwargs,
            )

        if not response.is_success:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response

    ######## MODELS ########
    @cached(
        cache=TTLCache(maxsize=1024, ttl=60 * 60 * 24),
        key=lambda self: hashkey(self._api_key),
    )
    async def list_models(self) -> list[dict]:
        """
        Returns the id for every model visible to the user.
        Cache: 1 day - per user
        """
        response = await self._request("GET", "/models")
        return response.json()["data"]

    @cached(cache=TTLCache(maxsize=1, ttl=60 * 60 * 24))
    async def get_model_info(self) -> list[dict]:
        """
        Returns detailed info (capabilities, cost, limits) for every model
        visible to the current API key. Users do not have access to this endpoint,
        so we use the service API key.
        Cache: 1 day - global
        """
        response = await self._request(
            "GET", "/model/info", api_key=settings.litellm_api_key
        )
        return response.json()["data"]

    @cached(cache=TTLCache(maxsize=1, ttl=60 * 60 * 24 * 7))
    async def get_models_status(self) -> list[dict]:
        """
        Returns the status for every model visible to the current API key.
        Users do not have access to this endpoint, so use the service API key.
        Cache: 1 week - global
        """
        response = await self._request(
            "GET",
            "/health/latest",
            api_key=settings.litellm_api_key,
        )
        return response.json()["latest_health_checks"]

    async def get_model_capabilities(self, model_id: str) -> dict:
        """
        Returns the capabilities for a specific model.
        """
        all_model_info = await self.get_model_info()
        info_by_model = {
            m["model_name"]: m.get("model_info") or {} for m in all_model_info
        }
        return info_by_model.get(model_id, {})

    async def chat_completion(self, model: str, messages: list[dict]) -> dict:
        """(OLD) Non-streaming version (kept in case it's needed somewhere)."""
        response = await self._request(
            "POST",
            "/chat/completions",
            json={"model": model, "messages": messages},
        )
        return response.json()

    async def stream_chat_completion(
        self,
        model_id: str,
        messages: list[dict],
        collection_ids: list[str] | None = None,
    ):
        """
        An asynchronous generator that produces chunks of text from the
        LiteLLM streaming endpoint (SSE, OpenAI-compatible format).

        `collection_ids`: knowledge bases selected by the user for this
        message only. When present, the Milvus MCP tool is attached to the
        request and a system message tells the model which collections to
        search. The caller must have validated that the user can
        access them and that the model supports tool calling.

        Yields tuples `(channel, text)` where `channel` is:
        - "content": the actual answer.
        - "thinking": the model's chain of thought. Not every model/provider sends this field.
        - "status": "searching", emitted once when the model calls the MCP tool
        """
        payload = {"model": model_id, "messages": messages, "stream": True}
        if collection_ids:
            payload["messages"] = [
                self._collections_system_message(collection_ids),
                *messages,
            ]
            payload["tools"] = [self._milvus_mcp_tool()]
            payload["tool_choice"] = "auto"
        announced_search = False
        # timeout=None: a streaming response may take quite a long time to
        # complete; httpx’s default timeout would cut it short prematurely.
        headers = self._get_headers()
        async with httpx.AsyncClient(
            base_url=settings.litellm_base_url, timeout=None
        ) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                headers=headers,
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
                    choices = chunk.get("choices") or [{}]
                    delta = choices[0].get("delta", {})

                    # LiteLLM adds MCP info under provider_specific_fields.
                    extras = delta.get("provider_specific_fields") or {}
                    if extras.get("mcp_tool_calls") and not announced_search:
                        announced_search = True
                        yield "status", "searching"

                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield "thinking", reasoning

                    content = delta.get("content")
                    if content:
                        yield "content", content

    ######## AGENTS ########
    async def list_agents(self) -> list[dict]:
        """Returns all the agents configured in LiteLLM."""
        response = await self._request("GET", "/v1/agents")
        return response.json()

    async def stream_agent_run(self, chat_id: str, agent_id: str, messages: list[dict]):
        import urllib.parse
        from uuid import uuid4

        parsed_url = urllib.parse.urlparse(settings.litellm_base_url)
        clean_base = f"{parsed_url.scheme}://{parsed_url.netloc}"
        base_url = f"{clean_base}/a2a/{agent_id}"
        headers = self._get_headers()

        last_content = messages[-1]["content"] if messages else ""
        previous_messages = messages[:-1] if messages else ""

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

            # LiteLLM payload
            request_id = str(uuid4())
            message_id = uuid4().hex

            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "message/stream",
                "params": {
                    "message": {
                        "kind": "message",
                        "messageId": message_id,
                        "contextId": chat_id,
                        "role": "user",
                        "parts": [{"kind": "text", "text": last_content}],
                        "metadata": {
                            "conversation_history": previous_messages,
                        },
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
                                artifact = result["artifact"]
                                channel = (artifact.get("metadata") or {}).get(
                                    "channel"
                                )
                                mapped_channel = (
                                    "thinking" if channel == "thinking" else "content"
                                )

                                for part in artifact.get("parts", []):
                                    if part.get("kind") == "text":
                                        text = part.get("text")
                                        if text:
                                            yield mapped_channel, text

                            elif "message" in result:
                                for part in result["message"].get("parts", []):
                                    if part.get("kind") == "text":
                                        text = part.get("text")
                                        if text:
                                            yield "content", text

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
