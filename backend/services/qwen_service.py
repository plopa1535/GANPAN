"""
Qwen LLM Service (via Groq)
스토리/스크립트 생성을 위한 서비스
"""

import os
from typing import Dict, List, Any
import httpx
from dotenv import load_dotenv

load_dotenv()


class QwenService:
    """Groq API를 통한 Qwen 스크립트 생성 서비스"""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        # Groq API (OpenAI 호환)
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        # Groq에서 사용 가능한 Qwen 모델 (2025년 기준)
        self.model = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

    async def generate_script(
        self,
        image_analyses: List[Dict[str, Any]],
        user_story: str,
        style: str = "cinematic",
        music_mood: str = "calm"
    ) -> str:
        """
        이미지 분석 결과와 사용자 스토리를 바탕으로 영상 스크립트 생성

        Args:
            image_analyses: 이미지 분석 결과 리스트
            user_story: 사용자가 입력한 스토리
            style: 영상 스타일 (cinematic, emotional, dynamic, minimal)
            music_mood: 배경음악 분위기 (calm, upbeat, epic, nostalgic)

        Returns:
            생성된 스크립트 텍스트
        """
        if not self.api_key:
            # API 키가 없으면 더미 스크립트 반환 (개발용)
            return self._get_dummy_script(image_analyses, user_story, style, music_mood)

        # 프롬프트 구성
        prompt = self._build_prompt(image_analyses, user_story, style, music_mood)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": self._get_system_prompt()
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.7,
                        "top_p": 0.9
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()

            # OpenAI 형식 응답에서 텍스트 추출
            choices = result.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                return text if text else self._get_dummy_script(image_analyses, user_story, style, music_mood)

            return self._get_dummy_script(image_analyses, user_story, style, music_mood)

        except Exception as e:
            print(f"Groq API error: {e}")
            return self._get_dummy_script(image_analyses, user_story, style, music_mood)

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 반환"""
        return """당신은 전문 영상 스크립트 작가입니다.
사용자가 제공한 이미지 분석 결과와 스토리를 바탕으로 1분짜리 쇼츠 영상을 위한 감동적이고 몰입감 있는 스크립트를 작성합니다.

스크립트 작성 가이드라인:
1. 총 길이는 약 60초 분량 (나레이션 기준 약 150-180단어)
2. 시각적 지시와 나레이션을 포함
3. 감정의 흐름을 자연스럽게 구성
4. 사용자의 스토리 의도를 존중하면서 창의적으로 발전
5. 이미지의 순서를 고려한 시퀀스 구성

출력 형식:
[장면 1 - 시간]
비주얼: (화면에 보여질 내용)
나레이션: (음성 나레이션)
효과: (전환 효과, 음악 변화 등)

[장면 2 - 시간]
...
"""

    def _build_prompt(
        self,
        image_analyses: List[Dict[str, Any]],
        user_story: str,
        style: str,
        music_mood: str
    ) -> str:
        """사용자 프롬프트 구성"""
        # 이미지 분석 요약
        image_descriptions = []
        for i, analysis in enumerate(image_analyses, 1):
            desc = f"이미지 {i}: {analysis.get('description', '분석 없음')}"
            labels = [l['description'] for l in analysis.get('labels', [])[:3]]
            if labels:
                desc += f" (키워드: {', '.join(labels)})"
            emotions = analysis.get('emotions', [])
            if emotions and emotions[0] != 'neutral':
                desc += f" (감정: {emotions[0]})"
            image_descriptions.append(desc)

        # 스타일 설명
        style_descriptions = {
            "cinematic": "영화같은 드라마틱한 연출, 깊은 감정선",
            "emotional": "감성적이고 따뜻한 톤, 잔잔한 감동",
            "dynamic": "빠른 전환, 에너지 넘치는 편집",
            "minimal": "미니멀하고 세련된 연출, 여백의 미"
        }

        music_descriptions = {
            "calm": "잔잔하고 평화로운 피아노/어쿠스틱",
            "upbeat": "밝고 경쾌한 팝/일렉트로닉",
            "epic": "웅장하고 드라마틱한 오케스트라",
            "nostalgic": "향수를 불러일으키는 레트로/어쿠스틱"
        }

        prompt = f"""## 이미지 분석 결과
{chr(10).join(image_descriptions)}

## 사용자 스토리
{user_story}

## 영상 스타일
{style_descriptions.get(style, style)}

## 배경음악 분위기
{music_descriptions.get(music_mood, music_mood)}

위 정보를 바탕으로 1분짜리 쇼츠 영상 스크립트를 작성해주세요.
총 {len(image_analyses)}개의 이미지를 순서대로 활용해주세요.
"""

        return prompt

    def _get_dummy_script(
        self,
        image_analyses: List[Dict[str, Any]],
        user_story: str,
        style: str,
        music_mood: str
    ) -> str:
        """개발/테스트용 더미 스크립트"""
        num_images = len(image_analyses)
        scene_duration = 60 // max(num_images, 1)

        script_parts = [
            f"[스크립트 - {style} 스타일 / {music_mood} 분위기]\n",
            f"원본 스토리: {user_story}\n",
            "=" * 40 + "\n"
        ]

        for i in range(num_images):
            start_time = i * scene_duration
            end_time = (i + 1) * scene_duration

            analysis = image_analyses[i] if i < len(image_analyses) else {}
            labels = [l['description'] for l in analysis.get('labels', [])[:2]]
            label_text = ', '.join(labels) if labels else '순간'

            script_parts.append(f"""
[장면 {i + 1}] {start_time}s - {end_time}s
비주얼: 이미지 {i + 1} - {label_text}의 장면이 펼쳐집니다
나레이션: 이 순간을 기억하세요, 우리의 이야기가 담긴 소중한 시간입니다.
효과: {'페이드 인' if i == 0 else '부드러운 전환'}
""")

        script_parts.append(f"""
[엔딩] 55s - 60s
비주얼: 모든 이미지가 콜라주로 모이며 마무리
나레이션: 이것이 우리의 이야기입니다.
효과: 페이드 아웃, {music_mood} 음악 마무리
""")

        return "".join(script_parts)
