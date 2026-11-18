import json

import httpx
import pytest
from fastapi import HTTPException

from genai_api.services.litellm_client import LiteLLMClient

INVALID_KEY_BODY = {
    "error": {"message": "x", "type": "token_not_found_in_db", "code": "401"}
}
OTHER_401_BODY = {"error": {"message": "x", "type": "auth_error", "code": "401"}}


def mock_httpx(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def make_client(api_key: str, calls: list):
    async def on_invalid_key(failed: str) -> str:
        calls.append(failed)
        return "new-key"

    return LiteLLMClient(api_key=api_key, on_invalid_key=on_invalid_key)


@pytest.mark.asyncio
async def test_request_retries_once_with_new_key(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["authorization"] == "Bearer new-key":
            return httpx.Response(200, json={"data": [{"id": "m1"}]})
        return httpx.Response(401, json=INVALID_KEY_BODY)

    mock_httpx(monkeypatch, handler)
    calls: list = []
    client = make_client("old-key-1", calls)

    assert await client.list_models() == [{"id": "m1"}]
    assert calls == ["old-key-1"]


@pytest.mark.asyncio
async def test_request_second_401_becomes_502(monkeypatch):
    mock_httpx(monkeypatch, lambda r: httpx.Response(401, json=INVALID_KEY_BODY))
    calls: list = []
    client = make_client("old-key-2", calls)

    with pytest.raises(HTTPException) as exc:
        await client.list_models()
    assert exc.value.status_code == 502
    assert len(calls) == 1  # only one refresh, no loop


@pytest.mark.asyncio
async def test_other_401_is_not_retried(monkeypatch):
    mock_httpx(monkeypatch, lambda r: httpx.Response(401, json=OTHER_401_BODY))
    calls: list = []
    client = make_client("old-key-3", calls)

    with pytest.raises(HTTPException) as exc:
        await client.list_models()
    assert exc.value.status_code == 502
    assert calls == []


@pytest.mark.asyncio
async def test_stream_retries_before_first_chunk(monkeypatch):
    sse = 'data: {"choices":[{"delta":{"content":"hola"}}]}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["authorization"] == "Bearer new-key":
            return httpx.Response(200, content=sse.encode())
        return httpx.Response(401, content=json.dumps(INVALID_KEY_BODY).encode())

    mock_httpx(monkeypatch, handler)
    calls: list = []
    client = make_client("old-key-4", calls)

    chunks = [
        c
        async for c in client.stream_chat_completion(
            "m", [{"role": "user", "content": "x"}]
        )
    ]
    assert chunks == [("content", "hola")]
    assert calls == ["old-key-4"]
