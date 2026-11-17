import json
import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from genai_api.main import app

TOKEN = os.getenv("TEST_USER_TOKEN")
MODEL_ID = os.getenv("TEST_MODEL_ID")
TEST_FILES_DIR = Path(__file__).parent / "test_files"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (TOKEN and MODEL_ID),
        reason="TEST_USER_TOKEN and TEST_MODEL_ID are required",
    ),
]


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:") :].strip())
        if event:
            events.append((event, data))
    return events


async def ask(client: AsyncClient, chat_id: str, content: str, files=None) -> str:
    """Sends a message, checks the stream ended well and returns the answer."""
    payload = {"content": content, "model_id": MODEL_ID}
    if files:
        payload["files"] = files
    res = await client.post(f"/chats/{chat_id}/messages", json=payload)
    assert res.status_code == 200, res.text
    events = parse_sse(res.text)
    names = [name for name, _ in events]
    assert "error" not in names, events
    assert names[-1] == "done"
    return "".join(
        d["content"] for name, d in events if name == "delta" and "content" in d
    )


@pytest.mark.asyncio
async def test_knowledge_chat():
    transport = ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    collection_id = None

    async with AsyncClient(
        transport=transport, base_url="http://test", headers=headers, timeout=180
    ) as client:
        try:
            # 1. Chat + its collection (type conversation, linked by chat id)
            res = await client.post(
                "/chats", json={"title": "test_knowledge_chat", "model_id": MODEL_ID}
            )
            assert res.status_code == 201, res.text
            chat_id = res.json()["id"]

            res = await client.post(
                "/knowledge/collections",
                json={
                    "name": "test_knowledge_chat",
                    "type": "conversation",
                    "conversation_id": chat_id,
                },
            )
            assert res.status_code == 201, res.text
            collection_id = res.json()["id"]

            # 2. Upload the mock CSV (Josefina, 78) to that collection
            csv_bytes = (TEST_FILES_DIR / "test.csv").read_bytes()
            res = await client.post(
                f"/knowledge/collections/{collection_id}/documents",
                files={"file": ("test.csv", csv_bytes)},
            )
            assert res.status_code == 200, res.text
            assert res.json()["status"] == "indexed"

            # 3. First message references the file
            answer = await ask(
                client,
                chat_id,
                "What is the age of Josefina?",
                files=[{"filename": "test.csv", "mimetype": "text/csv"}],
            )
            assert "78" in answer

            # 4. The reference was stored (name + mimetype only)
            res = await client.get(f"/chats/{chat_id}")
            assert res.status_code == 200
            first_user_message = res.json()["messages"][0]
            assert first_user_message["files"] == [
                {"filename": "test.csv", "mimetype": "text/csv"}
            ]

            # 5. Follow-up without files: the chat still has them, so it must
            #    keep querying the collection
            answer = await ask(client, chat_id, "And how old is Josefina, again?")
            assert "78" in answer
        finally:
            if collection_id:
                await client.delete(f"/knowledge/collections/{collection_id}")


@pytest.mark.asyncio
async def test_files_without_collection_are_rejected():
    transport = ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    async with AsyncClient(
        transport=transport, base_url="http://test", headers=headers, timeout=60
    ) as client:
        res = await client.post("/chats", json={"model_id": MODEL_ID})
        assert res.status_code == 201, res.text
        chat_id = res.json()["id"]

        res = await client.post(
            f"/chats/{chat_id}/messages",
            json={
                "content": "hi",
                "model_id": MODEL_ID,
                "files": [{"filename": "test.csv"}],
            },
        )
        assert res.status_code == 400
