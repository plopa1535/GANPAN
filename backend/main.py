"""
GANPAN - AI 쇼츠 영상 제작 API
FastAPI Backend
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uuid
import os
from typing import List
from pathlib import Path

from api.routes import router as api_router

# Create FastAPI app
app = FastAPI(
    title="GANPAN API",
    description="AI 쇼츠 영상 제작 API - Google Vision + Qwen + Replicate",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용, 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for generated videos
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Include API routes
app.include_router(api_router, prefix="/api")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "GANPAN API"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "GANPAN API - AI 쇼츠 영상 제작기",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
