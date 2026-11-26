import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from genai_api.api.routes.agents import router as agents_router
from genai_api.api.routes.chats import router as chats_router
from genai_api.api.routes.knowledge import router as knowledge_router
from genai_api.api.routes.models import router as models_router
from genai_api.api.routes.user import router as user_router
from genai_api.core.config import settings

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(levelname)-5s [%(name)s] %(message)s")

app = FastAPI(title="GenAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
)

app.include_router(chats_router)
app.include_router(models_router)
app.include_router(agents_router)
app.include_router(knowledge_router)
app.include_router(user_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
