from fastapi import FastAPI

from genai_api.api.routes.chats import router as chats_router
from genai_api.api.routes.models import router as models_router
from genai_api.api.routes.agents import router as agents_router

app = FastAPI(title="GenAI API")

app.include_router(chats_router)
app.include_router(models_router)
app.include_router(agents_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
