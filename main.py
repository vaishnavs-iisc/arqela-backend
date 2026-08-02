"""
Application entry point.

Responsibilities:
  - Create the FastAPI app
  - Register middleware (CORS)
  - Manage database pool lifecycle (lifespan)
  - Register all API routers
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import db_manager
from routes.hypothesis import router as hypothesis_router
from routes.research import router as research_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising PostgreSQL connection pool...")
    try:
        db_manager.init_pool()
        db_manager.init_db()
    except Exception as e:
        logger.error(f"Postgres pool startup notice: {e}")
    yield
    logger.info("Shutting down — closing PostgreSQL connection pool...")
    try:
        db_manager.close_pool()
    except Exception as e:
        logger.error(f"Postgres pool shutdown notice: {e}")


app = FastAPI(
    title="Multi-Agent AI Research Platform API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hypothesis_router)
app.include_router(research_router)
