"""
Video Generation Service
이미지들을 슬라이드쇼 영상으로 변환 + AI 클립 병합
"""

import os
import asyncio
from typing import List, Optional, Callable
from pathlib import Path
from PIL import Image
import imageio
import numpy as np


class VideoService:
    """이미지 슬라이드쇼 영상 생성 서비스"""

    def __init__(self):
        # 메모리 절약을 위해 해상도와 FPS 낮춤
        self.target_size = (720, 1280)  # 720p 세로 영상 (메모리 효율)
        self.fps = 24  # 24fps로 낮춤 (메모리 절약)

    async def create_slideshow(
        self,
        image_paths: List[str],
        output_path: str,
        duration_per_image: float = 5.0,
        transition_duration: float = 1.0,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        이미지들로 슬라이드쇼 영상 생성 (스트리밍 방식)

        Args:
            image_paths: 이미지 파일 경로 리스트
            output_path: 출력 영상 경로
            duration_per_image: 이미지당 표시 시간 (초)
            transition_duration: 전환 효과 시간 (초)
            progress_callback: 진행률 콜백

        Returns:
            생성된 영상 경로
        """
        if not image_paths:
            raise ValueError("이미지가 없습니다")

        # 60초에 맞춰 이미지당 시간 계산
        total_duration = 60.0
        num_images = len(image_paths)
        duration_per_image = total_duration / num_images

        # 최소 2초, 최대 10초
        duration_per_image = max(2.0, min(10.0, duration_per_image))
        transition_duration = min(transition_duration, duration_per_image / 3)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._create_slideshow_streaming,
                image_paths,
                output_path,
                duration_per_image,
                transition_duration,
                progress_callback
            )

            if progress_callback:
                progress_callback(100)

            return output_path

        except Exception as e:
            print(f"Video creation error: {e}")
            raise

    def _create_slideshow_streaming(
        self,
        image_paths: List[str],
        output_path: str,
        duration_per_image: float,
        transition_duration: float,
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        """스트리밍 방식으로 슬라이드쇼 생성 - 메모리 효율적"""
        frames_per_image = int(duration_per_image * self.fps)
        transition_frames = int(transition_duration * self.fps)
        total_images = len(image_paths)

        # 바로 파일에 쓰기 시작
        writer = imageio.get_writer(
            output_path,
            fps=self.fps,
            codec='libx264',
            quality=7,
            pixelformat='yuv420p',
            macro_block_size=8
        )

        prev_img_array = None

        for img_idx, path in enumerate(image_paths):
            # 이미지 로드 및 처리
            img = self._process_image_sync(path)
            img_array = np.array(img)

            for frame_idx in range(frames_per_image):
                # Ken Burns 효과 (줌 인/아웃)
                progress = frame_idx / frames_per_image
                zoom = 1.0 + 0.1 * progress

                frame = self._apply_ken_burns(img_array, zoom, progress)

                # 페이드 인 (전환 구간)
                if frame_idx < transition_frames and prev_img_array is not None:
                    alpha = frame_idx / transition_frames
                    prev_frame = self._apply_ken_burns(prev_img_array, 1.1, 1.0)
                    frame = self._blend_frames(prev_frame, frame, alpha)

                writer.append_data(frame)

            prev_img_array = img_array

            # 진행률 업데이트
            if progress_callback:
                progress_callback((img_idx + 1) / total_images * 90)

        writer.close()
        print(f"Slideshow saved to {output_path}")

    def _process_image_sync(self, path: str) -> Image.Image:
        """동기 이미지 처리"""
        img = Image.open(path).convert("RGB")

        # 9:16 비율로 크롭/리사이즈
        target_ratio = self.target_size[0] / self.target_size[1]  # 0.5625
        img_ratio = img.width / img.height

        if img_ratio > target_ratio:
            # 이미지가 더 넓음 - 좌우 크롭
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            # 이미지가 더 높음 - 상하 크롭
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))

        # 타겟 사이즈로 리사이즈
        img = img.resize(self.target_size, Image.Resampling.LANCZOS)

        return img

    def _apply_ken_burns(self, img_array, zoom: float, progress: float):
        """Ken Burns 효과 적용"""
        import numpy as np
        from PIL import Image

        h, w = img_array.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)

        # 확대
        img = Image.fromarray(img_array)
        img_zoomed = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 중앙에서 살짝 이동하며 크롭 (팬 효과)
        pan_x = int((new_w - w) * 0.5 * (1 + 0.5 * progress))
        pan_y = int((new_h - h) * 0.5 * (1 + 0.3 * progress))

        cropped = img_zoomed.crop((pan_x, pan_y, pan_x + w, pan_y + h))

        return np.array(cropped)

    def _blend_frames(self, frame1, frame2, alpha: float):
        """두 프레임 블렌딩"""
        return (frame1 * (1 - alpha) + frame2 * alpha).astype(np.uint8)

    async def merge_video_clips(
        self,
        clip_paths: List[str],
        output_path: str,
        transition_duration: float = 0.5,
        target_duration: float = 60.0,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        여러 AI 영상 클립을 하나로 병합

        Args:
            clip_paths: 클립 파일 경로 리스트
            output_path: 출력 영상 경로
            transition_duration: 전환 효과 시간 (초)
            target_duration: 목표 영상 길이 (초)
            progress_callback: 진행률 콜백

        Returns:
            병합된 영상 경로
        """
        if not clip_paths:
            raise ValueError("클립이 없습니다")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._merge_clips_sync,
            clip_paths,
            output_path,
            transition_duration,
            target_duration,
            progress_callback
        )

    def _merge_clips_sync(
        self,
        clip_paths: List[str],
        output_path: str,
        transition_duration: float,
        target_duration: float,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """동기 클립 병합 - 스트리밍 방식으로 메모리 효율화"""
        clip_fps = self.fps
        transition_frames = int(transition_duration * clip_fps)
        target_frames = int(target_duration * clip_fps)
        total_clips = len(clip_paths)

        # 스트리밍 방식: 바로 파일에 쓰기
        writer = imageio.get_writer(
            output_path,
            fps=self.fps,
            codec='libx264',
            quality=7,
            pixelformat='yuv420p',
            macro_block_size=8  # 720p에 맞게 조정
        )

        frames_written = 0
        last_frame = None

        for clip_idx, clip_path in enumerate(clip_paths):
            try:
                print(f"Processing clip {clip_idx + 1}/{total_clips}: {clip_path}")

                reader = imageio.get_reader(clip_path)
                clip_frame_count = 0

                for frame in reader:
                    if frames_written >= target_frames:
                        break

                    frame_resized = self._resize_frame_to_target(frame)

                    # 첫 클립이 아니고 전환 구간이면 블렌딩
                    if clip_idx > 0 and clip_frame_count < transition_frames and last_frame is not None:
                        alpha = clip_frame_count / transition_frames
                        frame_resized = self._blend_frames(last_frame, frame_resized, alpha)

                    writer.append_data(frame_resized)
                    last_frame = frame_resized
                    frames_written += 1
                    clip_frame_count += 1

                reader.close()

                if progress_callback:
                    progress_callback((clip_idx + 1) / total_clips * 80)

                if frames_written >= target_frames:
                    break

            except Exception as e:
                print(f"Error processing clip {clip_path}: {e}")
                continue

        # 프레임이 부족하면 마지막 프레임 반복
        if last_frame is not None:
            while frames_written < target_frames:
                writer.append_data(last_frame)
                frames_written += 1

        writer.close()

        if progress_callback:
            progress_callback(100)

        print(f"Saved video with {frames_written} frames to {output_path}")
        return output_path

    def _resize_frame_to_target(self, frame: np.ndarray) -> np.ndarray:
        """프레임을 타겟 사이즈로 리사이즈"""
        img = Image.fromarray(frame)

        # 9:16 비율로 크롭/리사이즈
        target_ratio = self.target_size[0] / self.target_size[1]
        img_ratio = img.width / img.height

        if img_ratio > target_ratio:
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))

        img = img.resize(self.target_size, Image.Resampling.LANCZOS)

        return np.array(img)
