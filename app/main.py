from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predictions
from app.core.config import settings

app = FastAPI(
    title="Alzheimer's Disease Progress Tracking API",
    description="API for tracking AD progression over time",
    version="1.0.0",
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1")
# app.include_router(health.router, prefix="/api/v1")
