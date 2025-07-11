"""
Main FastAPI application for DocuMind AI Assistant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.config import settings
from app.database import create_tables
from app.api import router
from app.api.auth import router as auth_router
from app.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Starting DocuMind AI Assistant...")
    create_tables()
    print("Database tables created successfully")

    yield

    # Shutdown
    print("Shutting down DocuMind AI Assistant...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered document Q&A system with source citations",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(router, prefix="/api/v1", tags=["api"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to DocuMind AI Assistant",
        "version": settings.app_version,
        "description": "AI-powered document Q&A system",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/docs")
async def get_docs():
    """Get API documentation"""
    return {
        "message": "API Documentation",
        "swagger_ui": "/docs",
        "redoc": "/redoc",
        "openapi_json": "/openapi.json",
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, reload=settings.debug
    )
