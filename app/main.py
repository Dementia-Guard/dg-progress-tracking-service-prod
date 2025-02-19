from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import logger

app = FastAPI(
    title="Alzheimer's Disease Progress Tracking API",
    description="API for tracking AD progression over time",
    version="1.0.0",
)


@asynccontextmanager
async def lifespan():
    logger.info("API is starting up...")


@app.get("/")
async def health_check():
    return {"status": "healthy"}
