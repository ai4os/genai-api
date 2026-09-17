from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from genai_api.api.deps import CurrentUser, get_current_user
from genai_api.main import app

try:
    from tests.conftest import TEST_USER_ID
except ModuleNotFoundError:
    from conftest import TEST_USER_ID

TEST_FILES_DIR = Path(__file__).parent / "test_files"


@pytest.mark.asyncio
async def test_knowledge_lifecycle():
    fake_user = CurrentUser(
        id=TEST_USER_ID,
        email="test@example.com",
        username="test",
        roles=[],
        raw_token="fake-token",
    )
    app.dependency_overrides[get_current_user] = lambda: fake_user

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. List user collections
            initial_res = await client.get("/knowledge/collections")
            assert initial_res.status_code == 200
            initial_collections = initial_res.json()
            initial_ids = {c["id"] for c in initial_collections}

            # 2. Generate a new collection
            create_res = await client.post(
                "/knowledge/collections",
                json={
                    "name": "Test Collection",
                    "description": "A test knowledge collection",
                    "type": "knowledge_base",
                },
            )
            assert create_res.status_code == 201
            created_col = create_res.json()
            collection_id = created_col["id"]
            assert created_col["name"] == "Test Collection"
            assert created_col["type"] == "knowledge_base"

            # 3. Make sure it was not in the old collections
            assert collection_id not in initial_ids

            listed_res = await client.get("/knowledge/collections")
            assert listed_res.status_code == 200
            listed_ids = {c["id"] for c in listed_res.json()}
            assert collection_id in listed_ids

            # 4. Upload MD, CSV and PDF from test_files to the collection
            test_files = ["test.md", "test.csv", "test.pdf"]
            for filename in test_files:
                file_path = TEST_FILES_DIR / filename
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                upload_res = await client.post(
                    f"/knowledge/collections/{collection_id}/documents",
                    files={"file": (filename, file_bytes)},
                )
                assert upload_res.status_code == 200
                upload_data = upload_res.json()
                assert upload_data["filename"] == filename
                assert upload_data["collection_id"] == collection_id
                assert upload_data["chunks_count"] > 0
                assert upload_data["status"] == "indexed"

            # 5. List all documents in the collection
            docs_res = await client.get(
                f"/knowledge/collections/{collection_id}/documents"
            )
            assert docs_res.status_code == 200
            doc_filenames = docs_res.json()
            assert set(doc_filenames) == set(test_files)

            # 6. Delete a particular file within the collection
            file_to_delete = "test.csv"
            del_file_res = await client.delete(
                f"/knowledge/collections/{collection_id}/documents/{file_to_delete}"
            )
            assert del_file_res.status_code == 200
            del_file_data = del_file_res.json()
            assert del_file_data["filename"] == file_to_delete
            assert del_file_data["collection_id"] == collection_id

            # Verify the file is no longer in the collection
            docs_after_del_res = await client.get(
                f"/knowledge/collections/{collection_id}/documents"
            )
            assert docs_after_del_res.status_code == 200
            remaining_docs = docs_after_del_res.json()
            assert file_to_delete not in remaining_docs
            assert set(remaining_docs) == set(test_files) - {file_to_delete}

            # 7. Delete the collection
            delete_res = await client.delete(f"/knowledge/collections/{collection_id}")
            assert delete_res.status_code == 200
            delete_data = delete_res.json()
            assert delete_data["collection_id"] == collection_id

            # Verify collection is removed
            after_delete_res = await client.get("/knowledge/collections")
            assert after_delete_res.status_code == 200
            after_delete_ids = {c["id"] for c in after_delete_res.json()}
            assert collection_id not in after_delete_ids
    finally:
        app.dependency_overrides.clear()


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_knowledge_lifecycle())