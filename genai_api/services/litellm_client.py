"""
Client for interacting with the LiteLLM API.

The client uses the user's API key for all requests by default.

Endpoints that require administrative privileges (e.g. health/status)
explicitly use the service API key from settings.
"""

import json
import logging
import urllib.parse
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver
from asyncache import cached
from cachetools import TTLCache
from cachetools.keys import hashkey
from fastapi import HTTPException

from genai_api.core.config import settings

logger = logging.getLogger(__name__)

# Alias of the Milvus MCP server registered in LiteLLM and its tools
MILVUS_MCP_SERVER = "knowledge_milvus"
MILVUS_MCP_TOOL = "search"


class LiteLLMClient:
    """One instance per request (see `get_user_litellm_client`).

    Because instances are short-lived, any `@cached` method MUST define `key=`
    explicitly: the default key includes `self`, so a new instance never hits the cache.
    """

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
            logger.warning(
                "LiteLLM %s %s answered %s", method, path, response.status_code
            )
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

    @cached(cache=TTLCache(maxsize=1, ttl=60 * 60 * 24), key=lambda self: hashkey())
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

    @cached(cache=TTLCache(maxsize=1, ttl=60 * 60 * 24 * 7), key=lambda self: hashkey())
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
                    logger.warning(
                        "LiteLLM chat/completions answered %s", response.status_code
                    )
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

    @staticmethod
    def _agent_chunk_to_events(result: dict) -> list[tuple[str, str]] | None:
        """Text events carried by one A2A chunk, or None if the chunk is not a
        content chunk (e.g. a status-update)."""
        if result.get("kind") == "artifact-update" and "artifact" in result:
            artifact = result["artifact"]
            channel = (artifact.get("metadata") or {}).get("channel")
            mapped = "thinking" if channel == "thinking" else "content"
            parts = artifact.get("parts", [])
        elif "message" in result:
            mapped = "content"
            parts = result["message"].get("parts", [])
        else:
            return None
        return [
            (mapped, part["text"])
            for part in parts
            if part.get("kind") == "text" and part.get("text")
        ]

    async def _resolve_agent_url(
        self, http_client: httpx.AsyncClient, base_url: str
    ) -> str:
        """URL where the agent receives messages, read from its Agent Card."""
        try:
            card = await A2ACardResolver(
                httpx_client=http_client, base_url=base_url
            ).get_agent_card()
            target_url = card.supported_interfaces[0].url
        except Exception as exc:
            logger.exception("Could not resolve the agent card at %s", base_url)
            raise HTTPException(
                status_code=502, detail="Could not resolve the agent card"
            ) from exc
        # Force HTTPS: otherwise the server sends the request over HTTP, since it
        # is treated as a local request between machines on the same network.
        if target_url.startswith("http://"):
            target_url = target_url.replace("http://", "https://", 1)
        return target_url

    async def stream_agent_run(self, chat_id: str, agent_id: str, messages: list[dict]):
        """
        Sends the conversation to an A2A agent (through LiteLLM) and yields
        `(channel, text)` tuples, like `stream_chat_completion`, plus
        ("heartbeat", "") while the agent is working.

        The last message is the question; the previous ones travel as
        `conversation_history` in the message metadata.
        """
        parsed_url = urllib.parse.urlparse(settings.litellm_base_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/a2a/{agent_id}"

        last_content = messages[-1]["content"] if messages else ""
        previous_messages = messages[:-1]

        async with httpx.AsyncClient(
            headers=self._get_headers(), timeout=None
        ) as http_client:
            target_url = await self._resolve_agent_url(http_client, base_url)

            payload = {
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "message/stream",
                "params": {
                    "message": {
                        "kind": "message",
                        "messageId": uuid4().hex,
                        "contextId": chat_id,
                        "role": "user",
                        "parts": [{"kind": "text", "text": last_content}],
                        "metadata": {"conversation_history": previous_messages},
                    }
                },
            }

            # Only network errors are translated to a 502; the HTTPException we
            # raise ourselves (agent answered 4xx/5xx) keeps its real status code.
            try:
                async with http_client.stream(
                    "POST", target_url, json=payload
                ) as response:
                    if not response.is_success:
                        body = await response.aread()
                        logger.warning(
                            "Agent %s answered %s", agent_id, response.status_code
                        )
                        raise HTTPException(
                            status_code=response.status_code, detail=body.decode()
                        )

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line in ("data: [DONE]", "[DONE]"):
                            break
                        if line.startswith("data:"):
                            line = line[len("data:") :].strip()

                        try:
                            result = json.loads(line).get("result", {})
                        except json.JSONDecodeError:
                            logger.debug(
                                "Ignoring non-JSON line from agent %s", agent_id
                            )
                            continue

                        events = self._agent_chunk_to_events(result)
                        if events is not None:
                            for event in events:
                                yield event
                        elif result.get("kind") == "status-update":
                            if result.get("final") is True:
                                break
                            # LiteLLM heartbeat: forwarded so the connection to
                            # the browser never goes silent.
                            yield "heartbeat", ""
            except httpx.HTTPError as exc:
                logger.exception("Connection error while talking to agent %s", agent_id)
                raise HTTPException(
                    status_code=502,
                    detail="Connection error while talking to the agent",
                ) from exc
