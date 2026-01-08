"""
Google Vision API Service
이미지 분석을 위한 서비스
"""

import os
import base64
from typing import Dict, List, Any
import httpx
from dotenv import load_dotenv

load_dotenv()


class VisionService:
    """Google Vision API를 사용한 이미지 분석 서비스"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_VISION_API_KEY")
        self.api_url = "https://vision.googleapis.com/v1/images:annotate"

    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        이미지를 분석하여 다양한 정보를 추출

        Args:
            image_path: 분석할 이미지 파일 경로

        Returns:
            분석 결과 딕셔너리
        """
        if not self.api_key:
            # API 키가 없으면 더미 데이터 반환 (개발용)
            return self._get_dummy_analysis(image_path)

        # 이미지를 base64로 인코딩
        with open(image_path, "rb") as f:
            image_content = base64.b64encode(f.read()).decode("utf-8")

        # API 요청 본문 구성
        request_body = {
            "requests": [
                {
                    "image": {"content": image_content},
                    "features": [
                        {"type": "LABEL_DETECTION", "maxResults": 10},
                        {"type": "FACE_DETECTION", "maxResults": 5},
                        {"type": "LANDMARK_DETECTION", "maxResults": 5},
                        {"type": "TEXT_DETECTION", "maxResults": 10},
                        {"type": "SAFE_SEARCH_DETECTION"},
                        {"type": "IMAGE_PROPERTIES"},
                        {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
                    ],
                }
            ]
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}?key={self.api_key}",
                    json=request_body,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

            return self._parse_response(result, image_path)

        except Exception as e:
            print(f"Vision API error: {e}")
            return self._get_dummy_analysis(image_path)

    def _parse_response(self, result: Dict, image_path: str) -> Dict[str, Any]:
        """API 응답을 파싱하여 구조화된 데이터로 변환"""
        response = result.get("responses", [{}])[0]

        analysis = {
            "image_path": image_path,
            "labels": [],
            "faces": [],
            "landmarks": [],
            "texts": [],
            "objects": [],
            "colors": [],
            "emotions": [],
            "description": ""
        }

        # 라벨 (이미지 내용 설명)
        for label in response.get("labelAnnotations", []):
            analysis["labels"].append({
                "description": label.get("description"),
                "score": label.get("score", 0)
            })

        # 얼굴 감지 및 감정
        for face in response.get("faceAnnotations", []):
            emotions = {
                "joy": face.get("joyLikelihood", "UNKNOWN"),
                "sorrow": face.get("sorrowLikelihood", "UNKNOWN"),
                "anger": face.get("angerLikelihood", "UNKNOWN"),
                "surprise": face.get("surpriseLikelihood", "UNKNOWN")
            }
            analysis["faces"].append(emotions)
            analysis["emotions"].append(self._get_dominant_emotion(emotions))

        # 랜드마크
        for landmark in response.get("landmarkAnnotations", []):
            analysis["landmarks"].append({
                "name": landmark.get("description"),
                "score": landmark.get("score", 0)
            })

        # 텍스트
        text_annotations = response.get("textAnnotations", [])
        if text_annotations:
            analysis["texts"] = [text_annotations[0].get("description", "")]

        # 객체
        for obj in response.get("localizedObjectAnnotations", []):
            analysis["objects"].append({
                "name": obj.get("name"),
                "score": obj.get("score", 0)
            })

        # 색상
        properties = response.get("imagePropertiesAnnotation", {})
        colors = properties.get("dominantColors", {}).get("colors", [])
        for color in colors[:5]:
            rgb = color.get("color", {})
            analysis["colors"].append({
                "r": rgb.get("red", 0),
                "g": rgb.get("green", 0),
                "b": rgb.get("blue", 0),
                "score": color.get("score", 0)
            })

        # 이미지 설명 생성
        analysis["description"] = self._generate_description(analysis)

        return analysis

    def _get_dominant_emotion(self, emotions: Dict) -> str:
        """가장 강한 감정을 찾아 반환"""
        likelihood_order = {
            "VERY_LIKELY": 5,
            "LIKELY": 4,
            "POSSIBLE": 3,
            "UNLIKELY": 2,
            "VERY_UNLIKELY": 1,
            "UNKNOWN": 0
        }

        max_emotion = "neutral"
        max_score = 0

        for emotion, likelihood in emotions.items():
            score = likelihood_order.get(likelihood, 0)
            if score > max_score:
                max_score = score
                max_emotion = emotion

        return max_emotion if max_score >= 3 else "neutral"

    def _generate_description(self, analysis: Dict) -> str:
        """분석 결과를 바탕으로 이미지 설명 생성"""
        parts = []

        # 주요 라벨
        labels = [l["description"] for l in analysis["labels"][:5]]
        if labels:
            parts.append(f"이미지에서 {', '.join(labels)}이(가) 감지되었습니다.")

        # 얼굴 및 감정
        if analysis["faces"]:
            face_count = len(analysis["faces"])
            emotions = [e for e in analysis["emotions"] if e != "neutral"]
            if emotions:
                parts.append(f"{face_count}명의 사람이 있으며, {', '.join(set(emotions))}의 감정이 느껴집니다.")
            else:
                parts.append(f"{face_count}명의 사람이 있습니다.")

        # 랜드마크
        if analysis["landmarks"]:
            landmark_names = [l["name"] for l in analysis["landmarks"]]
            parts.append(f"장소: {', '.join(landmark_names)}")

        # 객체
        objects = [o["name"] for o in analysis["objects"][:5]]
        if objects:
            parts.append(f"주요 객체: {', '.join(objects)}")

        return " ".join(parts) if parts else "이미지 분석 결과가 없습니다."

    def _get_dummy_analysis(self, image_path: str) -> Dict[str, Any]:
        """개발/테스트용 더미 분석 결과"""
        import random

        labels = [
            "nature", "outdoor", "sky", "travel", "people",
            "family", "beach", "mountain", "city", "food"
        ]
        emotions = ["joy", "neutral", "surprise"]

        return {
            "image_path": image_path,
            "labels": [
                {"description": random.choice(labels), "score": random.uniform(0.7, 0.99)}
                for _ in range(5)
            ],
            "faces": [{"joy": "LIKELY", "sorrow": "UNLIKELY", "anger": "UNLIKELY", "surprise": "POSSIBLE"}],
            "landmarks": [],
            "texts": [],
            "objects": [
                {"name": random.choice(["person", "tree", "building", "car"]), "score": random.uniform(0.7, 0.95)}
                for _ in range(3)
            ],
            "colors": [
                {"r": random.randint(0, 255), "g": random.randint(0, 255), "b": random.randint(0, 255), "score": 0.3}
                for _ in range(3)
            ],
            "emotions": [random.choice(emotions)],
            "description": f"테스트 이미지 분석 결과입니다. ({image_path})"
        }
