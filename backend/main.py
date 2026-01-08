"""
GANPAN - AI 쇼츠 영상 제작 API
FastAPI Backend + Frontend 통합 서빙
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Include API routes first
app.include_router(api_router, prefix="/api")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "GANPAN API"}

# Root - Serve frontend index.html
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>GANPAN API</h1><p>Frontend not found. Visit <a href='/docs'>/docs</a> for API.</p>")

# Static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
