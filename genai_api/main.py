from fastapi import FastAPI

from genai_api.api.routes.chats import router as chats_router

app = FastAPI(title="GenAI API")

app.include_router(chats_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
