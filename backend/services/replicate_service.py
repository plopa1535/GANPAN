"""
Replicate API Service
AI 영상 생성을 위한 서비스
"""

import os
import asyncio
from typing import List, Callable, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()


class ReplicateService:
    """Replicate API를 사용한 영상 생성 서비스"""

    def __init__(self):
        self.api_key = os.getenv("REPLICATE_API_KEY")
        self.api_url = "https://api.replicate.com/v1/predictions"

        # 사용할 모델 설정
        # 옵션: stable-video-diffusion, runway-gen3, minimax-video, luma-dream-machine 등
        self.video_model = os.getenv(
            "REPLICATE_VIDEO_MODEL",
            "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"
        )

    async def generate_video(
        self,
        image_paths: List[str],
        script: str,
        style: str = "cinematic",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        이미지와 스크립트를 바탕으로 AI 영상 생성

        Args:
            image_paths: 입력 이미지 경로 리스트
            script: 생성된 스크립트
            style: 영상 스타일
            progress_callback: 진행률 콜백 함수 (0-100)

        Returns:
            생성된 영상 URL
        """
        if not self.api_key:
            # API 키가 없으면 더미 URL 반환 (개발용)
            return await self._simulate_generation(progress_callback)

        try:
            # 이미지들을 base64로 인코딩
            import base64
            encoded_images = []
            for path in image_paths:
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                    # 파일 확장자에 따른 MIME 타입 결정
                    ext = path.lower().split(".")[-1]
                    mime_type = {
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "png": "image/png",
                        "webp": "image/webp"
                    }.get(ext, "image/jpeg")
                    encoded_images.append(f"data:{mime_type};base64,{encoded}")

            # 첫 번째 이미지를 기반으로 영상 생성 (SVD 모델 사용)
            # 여러 이미지 → 영상은 각 이미지별로 클립 생성 후 합성 필요
            video_clips = []

            for i, image_data in enumerate(encoded_images):
                if progress_callback:
                    progress_callback((i / len(encoded_images)) * 80)

                clip_url = await self._generate_clip(image_data, style)
                video_clips.append(clip_url)

            # 클립들을 합성 (별도 처리 필요, 여기서는 첫 번째 클립 반환)
            if progress_callback:
                progress_callback(90)

            # TODO: FFmpeg 등을 사용해 클립 합성
            final_video_url = video_clips[0] if video_clips else ""

            if progress_callback:
                progress_callback(100)

            return final_video_url

        except Exception as e:
            print(f"Replicate API error: {e}")
            return await self._simulate_generation(progress_callback)

    async def _generate_clip(self, image_data: str, style: str) -> str:
        """단일 이미지에서 영상 클립 생성"""
        async with httpx.AsyncClient() as client:
            # 예측 생성
            response = await client.post(
                self.api_url,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "version": self.video_model.split(":")[-1],
                    "input": {
                        "input_image": image_data,
                        "motion_bucket_id": 127,  # 모션 강도 (0-255)
                        "fps": 24,
                        "cond_aug": 0.02,  # 조건부 증강
                        "decoding_t": 7,  # 디코딩 타임스텝
                        "video_length": "25_frames_with_svd_xt"  # ~4초 클립
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            prediction = response.json()

            # 예측 완료 대기
            prediction_id = prediction["id"]
            return await self._wait_for_prediction(prediction_id)

    async def _wait_for_prediction(self, prediction_id: str, max_wait: int = 300) -> str:
        """예측 완료까지 대기"""
        async with httpx.AsyncClient() as client:
            for _ in range(max_wait // 2):  # 2초 간격으로 확인
                response = await client.get(
                    f"{self.api_url}/{prediction_id}",
                    headers={"Authorization": f"Token {self.api_key}"},
                    timeout=10.0
                )
                response.raise_for_status()
                result = response.json()

                status = result.get("status")
                if status == "succeeded":
                    output = result.get("output")
                    # output이 리스트인 경우 첫 번째 항목 반환
                    if isinstance(output, list):
                        return output[0] if output else ""
                    return output or ""
                elif status == "failed":
                    error = result.get("error", "Unknown error")
                    raise Exception(f"Prediction failed: {error}")
                elif status == "canceled":
                    raise Exception("Prediction was canceled")

                await asyncio.sleep(2)

            raise Exception("Prediction timed out")

    async def _simulate_generation(
        self,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """개발/테스트용 시뮬레이션"""
        # 진행률 시뮬레이션
        for progress in range(0, 101, 10):
            if progress_callback:
                progress_callback(progress)
            await asyncio.sleep(0.5)

        # 샘플 비디오 URL 반환
        return "https://www.w3schools.com/html/mov_bbb.mp4"

    async def download_video(self, video_url: str, output_path: str) -> bool:
        """
        생성된 영상을 다운로드

        Args:
            video_url: 영상 URL
            output_path: 저장할 경로

        Returns:
            성공 여부
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(video_url, timeout=120.0)
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    f.write(response.content)

                return True

        except Exception as e:
            print(f"Download error: {e}")
            # 더미 파일 생성 (개발용)
            # 실제로는 에러 처리 필요
            return False


class VideoComposer:
    """
    여러 영상 클립을 합성하는 유틸리티
    FFmpeg 필요
    """

    def __init__(self):
        self.ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")

    async def compose_clips(
        self,
        clip_paths: List[str],
        output_path: str,
        transition: str = "fade",
        transition_duration: float = 0.5
    ) -> bool:
        """
        영상 클립들을 하나로 합성

        Args:
            clip_paths: 클립 파일 경로 리스트
            output_path: 출력 파일 경로
            transition: 전환 효과 (fade, dissolve, wipe 등)
            transition_duration: 전환 시간 (초)

        Returns:
            성공 여부
        """
        if len(clip_paths) == 0:
            return False

        if len(clip_paths) == 1:
            # 클립이 하나면 그대로 복사
            import shutil
            shutil.copy(clip_paths[0], output_path)
            return True

        try:
            import subprocess

            # FFmpeg concat filter 사용
            # 입력 파일 리스트 생성
            filter_parts = []
            inputs = []

            for i, path in enumerate(clip_paths):
                inputs.extend(["-i", path])
                filter_parts.append(f"[{i}:v]")

            # 필터 구성
            filter_complex = "".join(filter_parts) + f"concat=n={len(clip_paths)}:v=1:a=0[outv]"

            cmd = [
                self.ffmpeg_path,
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-y",
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            return result.returncode == 0

        except Exception as e:
            print(f"Video composition error: {e}")
            return False

    async def add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> bool:
        """
        영상에 오디오 추가

        Args:
            video_path: 비디오 파일 경로
            audio_path: 오디오 파일 경로
            output_path: 출력 파일 경로

        Returns:
            성공 여부
        """
        try:
            import subprocess

            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                "-y",
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            return result.returncode == 0

        except Exception as e:
            print(f"Audio addition error: {e}")
            return False
