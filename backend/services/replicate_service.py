"""
Replicate API Service
AI 영상 생성을 위한 서비스
"""

import os
import asyncio
from typing import List, Callable, Optional
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()


class ReplicateService:
    """Replicate API를 사용한 영상 생성 서비스"""

    def __init__(self):
        self.api_key = os.getenv("REPLICATE_API_KEY")
        self.api_url = "https://api.replicate.com/v1/predictions"

        # 디버그: API 키 상태 출력
        if self.api_key:
            masked_key = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "***"
            print(f"Replicate API key loaded: {masked_key} (length: {len(self.api_key)})")
        else:
            print("WARNING: REPLICATE_API_KEY not found in environment")

        # 사용할 모델 설정
        self.video_model = os.getenv(
            "REPLICATE_VIDEO_MODEL",
            "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"
        )

    async def generate_video_clips(
        self,
        image_paths: List[str],
        output_dir: str,
        style: str = "cinematic",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[str]:
        """
        각 이미지에서 AI 영상 클립들을 병렬로 생성

        Args:
            image_paths: 입력 이미지 경로 리스트
            output_dir: 클립 저장 디렉토리
            style: 영상 스타일
            progress_callback: 진행률 콜백 함수 (0-100)

        Returns:
            생성된 영상 클립 경로 리스트 (순서 보장)
        """
        if not self.api_key:
            print("No Replicate API key - cannot generate AI clips")
            return []

        import base64

        total_images = len(image_paths)
        print(f"Starting parallel generation of {total_images} AI clips")

        if progress_callback:
            progress_callback(5)

        # Step 1: 모든 이미지를 base64로 인코딩
        encoded_images = []
        for i, path in enumerate(image_paths):
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = path.lower().split(".")[-1]
                mime_type = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "webp": "image/webp"
                }.get(ext, "image/jpeg")
                encoded_images.append(f"data:{mime_type};base64,{encoded}")

        if progress_callback:
            progress_callback(10)

        # Step 2: 모든 클립 생성 요청을 병렬로 시작
        print("Submitting all clip generation requests in parallel...")
        prediction_ids = await self._submit_all_predictions(encoded_images, style)

        if progress_callback:
            progress_callback(20)

        if not prediction_ids:
            print("No predictions submitted successfully")
            return []

        # Step 3: 모든 예측 완료를 병렬로 대기
        print(f"Waiting for {len(prediction_ids)} predictions to complete...")
        clip_urls = await self._wait_all_predictions(prediction_ids, progress_callback)

        if progress_callback:
            progress_callback(80)

        # Step 4: 모든 클립을 병렬로 다운로드
        print("Downloading all clips in parallel...")
        clip_paths = await self._download_all_clips(clip_urls, output_dir)

        if progress_callback:
            progress_callback(100)

        print(f"Successfully generated {len(clip_paths)} clips")
        return clip_paths

    async def _submit_all_predictions(
        self,
        encoded_images: List[str],
        style: str
    ) -> List[Optional[str]]:
        """모든 이미지에 대한 예측 요청을 병렬로 제출"""
        async def submit_one(image_data: str, index: int) -> Optional[str]:
            try:
                async with httpx.AsyncClient() as client:
                    print(f"Submitting prediction {index + 1}/{len(encoded_images)}")
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
                                "motion_bucket_id": 127,
                                "fps": 24,
                                "cond_aug": 0.02,
                                "decoding_t": 7,
                                "video_length": "25_frames_with_svd_xt"
                            }
                        },
                        timeout=60.0
                    )

                    if response.status_code not in [200, 201]:
                        print(f"Prediction {index + 1} failed: {response.status_code}")
                        print(f"Response: {response.text}")
                        return None

                    prediction = response.json()
                    return prediction.get("id")

            except Exception as e:
                print(f"Error submitting prediction {index + 1}: {e}")
                return None

        # 모든 요청을 병렬로 실행
        tasks = [submit_one(img, i) for i, img in enumerate(encoded_images)]
        results = await asyncio.gather(*tasks)
        return results

    async def _wait_all_predictions(
        self,
        prediction_ids: List[Optional[str]],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Optional[str]]:
        """모든 예측 완료를 병렬로 대기"""
        async def wait_one(pred_id: Optional[str], index: int) -> Optional[str]:
            if not pred_id:
                return None
            try:
                return await self._wait_for_prediction(pred_id)
            except Exception as e:
                print(f"Error waiting for prediction {index + 1}: {e}")
                return None

        # 완료된 작업 수 추적
        completed = [0]
        total = len([p for p in prediction_ids if p])

        async def wait_with_progress(pred_id: Optional[str], index: int) -> Optional[str]:
            result = await wait_one(pred_id, index)
            completed[0] += 1
            if progress_callback and total > 0:
                # 20% ~ 80% 구간에서 진행률 업데이트
                progress = 20 + (completed[0] / total) * 60
                progress_callback(progress)
            return result

        tasks = [wait_with_progress(pid, i) for i, pid in enumerate(prediction_ids)]
        results = await asyncio.gather(*tasks)
        return results

    async def _download_all_clips(
        self,
        clip_urls: List[Optional[str]],
        output_dir: str
    ) -> List[str]:
        """모든 클립을 병렬로 다운로드"""
        async def download_one(url: Optional[str], index: int) -> Optional[str]:
            if not url:
                return None
            try:
                clip_path = os.path.join(output_dir, f"clip_{index:03d}.mp4")
                success = await self.download_video(url, clip_path)
                if success:
                    print(f"Downloaded clip {index + 1}")
                    return clip_path
                return None
            except Exception as e:
                print(f"Error downloading clip {index + 1}: {e}")
                return None

        tasks = [download_one(url, i) for i, url in enumerate(clip_urls)]
        results = await asyncio.gather(*tasks)

        # None 제거하고 순서 유지
        return [path for path in results if path]

    async def _generate_clip(self, image_data: str, style: str) -> str:
        """단일 이미지에서 영상 클립 생성"""
        async with httpx.AsyncClient() as client:
            # 예측 생성
            print(f"Calling Replicate API with model: {self.video_model}")
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

            # 에러 상세 로깅
            if response.status_code != 200 and response.status_code != 201:
                print(f"Replicate API response status: {response.status_code}")
                print(f"Replicate API response body: {response.text}")

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
