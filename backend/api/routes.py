"""
API Routes for GANPAN
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import List
import uuid
import os
import asyncio
from pathlib import Path

from services.vision_service import VisionService
from services.qwen_service import QwenService
from services.replicate_service import ReplicateService
from services.video_service import VideoService
from services.task_manager import TaskManager

router = APIRouter()

# Services
vision_service = VisionService()
qwen_service = QwenService()
replicate_service = ReplicateService()
video_service = VideoService()
task_manager = TaskManager()

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")


@router.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    story: str = Form(...),
    style: str = Form("cinematic"),
    music: str = Form("calm")
):
    """
    영상 생성 요청을 받아 백그라운드에서 처리
    """
    # Validate inputs
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="최소 1장의 이미지가 필요합니다")

    if len(images) > 10:
        raise HTTPException(status_code=400, detail="최대 10장까지 업로드할 수 있습니다")

    if not story.strip():
        raise HTTPException(status_code=400, detail="스토리를 입력해주세요")

    # Create task ID
    task_id = str(uuid.uuid4())

    # Save uploaded images
    image_paths = []
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        file_ext = os.path.splitext(image.filename)[1] or ".jpg"
        file_path = task_dir / f"image_{i}{file_ext}"

        content = await image.read()
        with open(file_path, "wb") as f:
            f.write(content)

        image_paths.append(str(file_path))

    # Initialize task
    task_manager.create_task(task_id, {
        "image_paths": image_paths,
        "story": story,
        "style": style,
        "music": music
    })

    # Start background processing
    background_tasks.add_task(
        process_video_generation,
        task_id,
        image_paths,
        story,
        style,
        music
    )

    return {"task_id": task_id, "status": "processing"}


async def process_video_generation(
    task_id: str,
    image_paths: List[str],
    story: str,
    style: str,
    music: str
):
    """
    영상 생성 파이프라인 실행
    """
    try:
        # Step 1: Analyze images with Google Vision (0-30%)
        task_manager.update_progress(task_id, 5, "이미지 분석 시작")

        image_analyses = []
        for i, path in enumerate(image_paths):
            progress = 5 + (25 * (i + 1) / len(image_paths))
            task_manager.update_progress(task_id, progress, f"이미지 {i+1}/{len(image_paths)} 분석 중")

            analysis = await vision_service.analyze_image(path)
            image_analyses.append(analysis)

        task_manager.update_progress(task_id, 30, "이미지 분석 완료")

        # Step 2: Generate script with Qwen (30-50%)
        task_manager.update_progress(task_id, 35, "스크립트 생성 중")

        script = await qwen_service.generate_script(
            image_analyses=image_analyses,
            user_story=story,
            style=style,
            music_mood=music
        )

        task_manager.update_progress(task_id, 50, "스크립트 생성 완료")

        # Step 3: Generate video (50-100%)
        task_manager.update_progress(task_id, 55, "AI 영상 클립 생성 시작")

        output_path = OUTPUT_DIR / f"{task_id}.mp4"
        clips_dir = UPLOAD_DIR / task_id / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        # Replicate API로 각 이미지에서 AI 영상 클립 생성
        clip_paths = await replicate_service.generate_video_clips(
            image_paths=image_paths,
            output_dir=str(clips_dir),
            style=style,
            progress_callback=lambda p: task_manager.update_progress(
                task_id, 55 + (p * 0.30), "AI 영상 클립 생성 중"
            )
        )

        task_manager.update_progress(task_id, 85, "영상 클립 병합 중")

        if clip_paths:
            # AI 클립들을 병합하여 60초 영상 생성
            await video_service.merge_video_clips(
                clip_paths=clip_paths,
                output_path=str(output_path),
                transition_duration=0.5,
                target_duration=60.0,
                progress_callback=lambda p: task_manager.update_progress(
                    task_id, 85 + (p * 0.15), "영상 클립 병합 중"
                )
            )
        else:
            # AI 클립 생성 실패 시 슬라이드쇼로 폴백
            task_manager.update_progress(task_id, 85, "슬라이드쇼 영상 생성 중")
            await video_service.create_slideshow(
                image_paths=image_paths,
                output_path=str(output_path),
                progress_callback=lambda p: task_manager.update_progress(
                    task_id, 85 + (p * 0.15), "슬라이드쇼 생성 중"
                )
            )

        task_manager.update_progress(task_id, 100, "완료")

        # Mark as completed
        task_manager.complete_task(task_id, {
            "video_url": f"/outputs/{task_id}.mp4",
            "script": script
        })

    except Exception as e:
        task_manager.fail_task(task_id, str(e))
        raise


@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """
    작업 진행 상황 조회
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    return task


@router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """
    작업 취소
    """
    success = task_manager.cancel_task(task_id)

    if not success:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    return {"status": "cancelled", "task_id": task_id}
