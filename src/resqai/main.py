import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from resqai.api.analyze import router as analyze_router
from resqai.api.v1.router import api_v1_router
from resqai.configs.settings import get_settings
from resqai.utils.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="ResQAI",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Frontend integration (local dev by default). Override via `RESQAI_CORS_ORIGINS`
    # as a comma-separated list.
    cors_env = (os.getenv("RESQAI_CORS_ORIGINS") or "").strip()
    origins = [o.strip() for o in cors_env.split(",") if o.strip()] or [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_v1_router, prefix="/api/v1")
    # Text-only streaming orchestration endpoint for frontend integration.
    app.include_router(analyze_router)
    return app


app = create_app()
