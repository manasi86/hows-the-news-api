"""FastAPI application entry point."""

import logging
from os import environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analyse import router as analyse_router
from api.routes.cost import router as cost_router
from api.routes.pricing import router as pricing_router
from api.routes.summarize import router as summarize_router

logging.basicConfig(
    level=environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="hows-the-news", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(summarize_router)
app.include_router(analyse_router)
app.include_router(pricing_router)
app.include_router(cost_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness check for the API.

    Returns:
        A JSON payload indicating the API is healthy.
    """
    return {"status": "ok"}
