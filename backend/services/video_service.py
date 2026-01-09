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
        self.target_size = (1080, 1920)  # 9:16 세로 영상 (쇼츠용)
        self.fps = 30

    async def create_slideshow(
        self,
        image_paths: List[str],
        output_path: str,
        duration_per_image: float = 5.0,
        transition_duration: float = 1.0,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        이미지들로 슬라이드쇼 영상 생성

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
            # 이미지 로드 및 처리
            processed_images = []
            for i, path in enumerate(image_paths):
                if progress_callback:
                    progress_callback((i / len(image_paths)) * 30)

                img = await self._load_and_process_image(path)
                processed_images.append(img)

            if progress_callback:
                progress_callback(30)

            # 프레임 생성
            frames = await self._generate_frames(
                processed_images,
                duration_per_image,
                transition_duration,
                progress_callback
            )

            if progress_callback:
                progress_callback(80)

            # 영상 저장
            await self._save_video(frames, output_path)

            if progress_callback:
                progress_callback(100)

            return output_path

        except Exception as e:
            print(f"Video creation error: {e}")
            raise

    async def _load_and_process_image(self, path: str) -> Image.Image:
        """이미지 로드 및 9:16 비율로 처리"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_image_sync, path)

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

    async def _generate_frames(
        self,
        images: List[Image.Image],
        duration_per_image: float,
        transition_duration: float,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List:
        """프레임 생성 (Ken Burns 효과 + 페이드 전환)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_frames_sync,
            images,
            duration_per_image,
            transition_duration,
            progress_callback
        )

    def _generate_frames_sync(
        self,
        images: List[Image.Image],
        duration_per_image: float,
        transition_duration: float,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List:
        """동기 프레임 생성"""
        import numpy as np

        frames = []
        frames_per_image = int(duration_per_image * self.fps)
        transition_frames = int(transition_duration * self.fps)

        total_images = len(images)

        for img_idx, img in enumerate(images):
            img_array = np.array(img)

            for frame_idx in range(frames_per_image):
                # Ken Burns 효과 (줌 인/아웃)
                progress = frame_idx / frames_per_image
                zoom = 1.0 + 0.1 * progress  # 1.0 -> 1.1 줌

                frame = self._apply_ken_burns(img_array, zoom, progress)

                # 페이드 인 (첫 이미지 시작 또는 전환 구간)
                if frame_idx < transition_frames:
                    alpha = frame_idx / transition_frames
                    if img_idx > 0:
                        # 이전 이미지와 블렌딩
                        prev_img = np.array(images[img_idx - 1])
                        prev_frame = self._apply_ken_burns(prev_img, 1.1, 1.0)
                        frame = self._blend_frames(prev_frame, frame, alpha)

                frames.append(frame)

            # 진행률 업데이트
            if progress_callback:
                progress_callback(30 + (img_idx / total_images) * 50)

        return frames

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
        import numpy as np
        return (frame1 * (1 - alpha) + frame2 * alpha).astype(np.uint8)

    async def _save_video(self, frames: List, output_path: str):
        """영상 파일 저장"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_video_sync, frames, output_path)

    def _save_video_sync(self, frames: List, output_path: str):
        """동기 영상 저장"""
        writer = imageio.get_writer(
            output_path,
            fps=self.fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p',
            macro_block_size=16
        )

        for frame in frames:
            writer.append_data(frame)

        writer.close()

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
        """동기 클립 병합"""
        all_frames = []
        clip_fps = self.fps
        transition_frames = int(transition_duration * clip_fps)

        total_clips = len(clip_paths)

        for clip_idx, clip_path in enumerate(clip_paths):
            try:
                print(f"Reading clip {clip_idx + 1}/{total_clips}: {clip_path}")

                # 클립 읽기
                reader = imageio.get_reader(clip_path)
                clip_frames = []

                for frame in reader:
                    # 9:16 비율로 리사이즈
                    frame_resized = self._resize_frame_to_target(frame)
                    clip_frames.append(frame_resized)

                reader.close()

                if not clip_frames:
                    continue

                # 첫 클립이 아니면 전환 효과 적용
                if clip_idx > 0 and all_frames and transition_frames > 0:
                    # 이전 클립 끝 프레임들과 현재 클립 시작 프레임들 블렌딩
                    for t in range(min(transition_frames, len(clip_frames))):
                        alpha = t / transition_frames
                        if len(all_frames) > 0:
                            prev_frame = all_frames[-1]
                            curr_frame = clip_frames[t]
                            blended = self._blend_frames(prev_frame, curr_frame, alpha)
                            all_frames.append(blended)
                    # 나머지 프레임 추가
                    all_frames.extend(clip_frames[transition_frames:])
                else:
                    all_frames.extend(clip_frames)

                if progress_callback:
                    progress_callback((clip_idx + 1) / total_clips * 80)

            except Exception as e:
                print(f"Error reading clip {clip_path}: {e}")
                continue

        if not all_frames:
            raise ValueError("병합할 프레임이 없습니다")

        # 목표 길이에 맞게 조정 (필요시 반복 또는 자르기)
        target_frames = int(target_duration * clip_fps)
        if len(all_frames) < target_frames:
            # 프레임 부족 - 마지막 클립 반복
            while len(all_frames) < target_frames:
                all_frames.append(all_frames[-1])
        elif len(all_frames) > target_frames:
            # 프레임 초과 - 자르기
            all_frames = all_frames[:target_frames]

        if progress_callback:
            progress_callback(90)

        # 영상 저장
        print(f"Saving merged video with {len(all_frames)} frames to {output_path}")
        self._save_video_sync(all_frames, output_path)

        if progress_callback:
            progress_callback(100)

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
