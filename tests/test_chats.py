import asyncio
import contextlib
import uuid

import pytest
from fastapi import HTTPException

from genai_api.api.deps import get_current_user_id
from genai_api.main import app
from genai_api.services import chat_service
from tests.conftest import OTHER_USER_ID

pytestmark = pytest.mark.asyncio

MODEL = "some-model"


async def new_chat(client, title="Test") -> str:
    r = await client.post("/chats", json={"model_id": MODEL, "title": title})
    assert r.status_code == 201
    return r.json()["id"]


async def send(client, chat_id, **payload):
    payload.setdefault("content", "hola")
    return await client.post(f"/chats/{chat_id}/messages", json=payload)


async def messages_of(client, chat_id) -> list[dict]:
    r = await client.get(f"/chats/{chat_id}")
    assert r.status_code == 200
    return r.json()["messages"]


# ---------- happy paths ----------


async def test_create_list_and_send_to_model_and_agent(client, fake_llm):
    chat_id = await new_chat(client)
    assert any(c["id"] == chat_id for c in (await client.get("/chats")).json())

    r = await send(client, chat_id, model_id=MODEL)
    assert r.status_code == 200
    assert "event: delta" in r.text and "event: done" in r.text
    assert len(await messages_of(client, chat_id)) == 2

    r = await send(client, chat_id, agent_id="easy_agent")
    assert "event: done" in r.text
    msgs = await messages_of(client, chat_id)
    assert len(msgs) == 4
    assert msgs[1]["model_id"] == MODEL and msgs[1]["agent_id"] is None
    assert msgs[3]["agent_id"] == "easy_agent" and msgs[3]["model_id"] is None
    # the agent received the whole history (user, assistant, user)
    assert [m["role"] for m in fake_llm.last_messages] == ["user", "assistant", "user"]


async def test_sending_a_message_moves_the_chat_to_the_top(client):
    old = await new_chat(client, "old")
    await new_chat(client, "newer")
    assert (await client.get("/chats")).json()[0]["title"] == "newer"

    await send(client, old, model_id=MODEL)

    assert (await client.get("/chats")).json()[0]["id"] == old


# ---------- validation ----------


@pytest.mark.parametrize(
    "target", [{}, {"model_id": MODEL, "agent_id": "a"}], ids=["none", "both"]
)
async def test_exactly_one_of_model_or_agent(client, target):
    chat_id = await new_chat(client)
    assert (await send(client, chat_id, **target)).status_code == 400


async def test_agent_rejects_images_files_and_collections(client):
    chat_id = await new_chat(client)
    image = {"mime_type": "image/png", "data": "AAAA"}
    for extra in ({"images": [image]}, {"collection_ids": ["c"]}):
        r = await send(client, chat_id, agent_id="a", **extra)
        assert r.status_code == 400


async def test_images_require_a_vision_model(client, fake_llm):
    chat_id = await new_chat(client)
    image = {"mime_type": "image/png", "data": "AAAA"}

    assert (
        await send(client, chat_id, model_id=MODEL, images=[image])
    ).status_code == 400

    fake_llm.capabilities = {"supports_vision": True}
    r = await send(client, chat_id, model_id=MODEL, images=[image])
    assert r.status_code == 200
    # only the message being answered carries the image bytes
    assert fake_llm.last_messages[-1]["content"][1]["type"] == "image_url"


async def test_chat_of_another_user_is_404(client):
    chat_id = await new_chat(client)
    app.dependency_overrides[get_current_user_id] = lambda: OTHER_USER_ID

    assert (await client.get(f"/chats/{chat_id}")).status_code == 404
    assert (await send(client, chat_id, model_id=MODEL)).status_code == 404
    assert (await client.delete(f"/chats/{chat_id}")).status_code == 404


# ---------- collections (Milvus mocked) ----------


async def test_collections_must_be_accessible(client, fake_llm, monkeypatch):
    fake_llm.capabilities = {"supports_function_calling": True}
    monkeypatch.setattr(chat_service.milvus, "get_milvus_client", lambda: object())
    chat_id = await new_chat(client)

    monkeypatch.setattr(chat_service.milvus, "user_can_read", lambda *a: False)
    r = await send(client, chat_id, model_id=MODEL, collection_ids=["col_x"])
    assert r.status_code == 404

    monkeypatch.setattr(chat_service.milvus, "user_can_read", lambda *a: True)
    r = await send(client, chat_id, model_id=MODEL, collection_ids=["col_x"])
    assert r.status_code == 200
    assert fake_llm.last_collection_ids == ["col_x"]


async def test_collections_require_tool_calling(client, fake_llm):
    chat_id = await new_chat(client)
    r = await send(client, chat_id, model_id=MODEL, collection_ids=["col_x"])
    assert r.status_code == 400


# ---------- what gets saved when things go wrong ----------


async def test_partial_reply_is_saved_if_the_model_fails_midway(client, fake_llm):
    fake_llm.chunks = ["parcial"]
    fake_llm.error = HTTPException(502, "boom")
    chat_id = await new_chat(client)

    r = await send(client, chat_id, model_id=MODEL)

    assert "event: error" in r.text and "event: done" not in r.text
    msgs = await messages_of(client, chat_id)
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "hola"),
        ("assistant", "parcial"),
    ]


async def test_nothing_is_saved_if_the_model_fails_before_any_text(client, fake_llm):
    fake_llm.chunks = []
    fake_llm.error = HTTPException(502, "boom")
    chat_id = await new_chat(client)

    r = await send(client, chat_id, model_id=MODEL)

    assert "event: error" in r.text
    assert [m["role"] for m in await messages_of(client, chat_id)] == ["user"]


async def test_partial_reply_is_saved_when_the_request_is_cancelled(client):
    """What happens when the user presses Stop: the generator is cancelled."""
    chat_id = await new_chat(client)

    async def slow_model():
        yield "content", "a medias"
        await asyncio.sleep(30)

    stream = chat_service.stream_reply(
        chat_id=uuid.UUID(chat_id),
        model_id=MODEL,
        agent_id=None,
        chunks=slow_model(),
    )

    async def consume():
        async for _ in stream:
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.3)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    msgs = await messages_of(client, chat_id)
    assert [(m["role"], m["content"]) for m in msgs] == [("assistant", "a medias")]


# ---------- delete ----------


async def test_delete_chat(client, monkeypatch):
    dropped = []
    monkeypatch.setattr(chat_service.milvus, "get_milvus_client", lambda: object())
    monkeypatch.setattr(
        chat_service.milvus,
        "delete_conversation_collections",
        lambda client, user_id, chat_id: dropped.append(str(chat_id)),
    )
    chat_id = await new_chat(client)
    await send(client, chat_id, model_id=MODEL)

    assert (await client.delete(f"/chats/{chat_id}")).status_code == 204
    assert dropped == [chat_id]
    assert (await client.get(f"/chats/{chat_id}")).status_code == 404


async def test_chat_is_kept_if_milvus_cleanup_fails(client, monkeypatch):
    def boom(*args):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(chat_service.milvus, "get_milvus_client", lambda: object())
    monkeypatch.setattr(chat_service.milvus, "delete_conversation_collections", boom)
    chat_id = await new_chat(client)

    assert (await client.delete(f"/chats/{chat_id}")).status_code == 502
    assert (await client.get(f"/chats/{chat_id}")).status_code == 200
