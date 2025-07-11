#!/usr/bin/env python3
"""
Startup script for DocuMind AI Assistant
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Server will be available at http://{settings.host}:{settings.port}")
    print(f"API documentation: http://{settings.host}:{settings.port}/docs")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
