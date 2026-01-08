# GANPAN - AI 쇼츠 영상 제작기

사진 여러 장과 서사를 입력하면 1분짜리 개인 쇼츠 영상을 자동으로 제작해주는 웹 애플리케이션입니다.

## 기술 스택

| 구분 | 기술 |
|------|------|
| **Frontend** | HTML/CSS/JavaScript (Material Design) |
| **Backend** | Python FastAPI |
| **이미지 분석** | Google Vision API |
| **스크립트 생성** | Qwen (Alibaba Cloud DashScope) |
| **영상 생성** | Replicate (Stable Video Diffusion) |

## 프로젝트 구조

```
GANPAN/
├── frontend/
│   ├── index.html          # 메인 페이지
│   ├── css/
│   │   └── style.css       # Material Design 스타일
│   └── js/
│       └── app.js          # 프론트엔드 로직
├── backend/
│   ├── main.py             # FastAPI 앱 진입점
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py       # API 라우터
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vision_service.py    # Google Vision 연동
│   │   ├── qwen_service.py      # Qwen LLM 연동
│   │   ├── replicate_service.py # Replicate 영상 생성
│   │   └── task_manager.py      # 비동기 작업 관리
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

## 설치 및 실행

### 1. 백엔드 설정

```bash
# 백엔드 폴더로 이동
cd backend

# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 API 키 입력
```

### 2. API 키 발급

#### Google Vision API
1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. 새 프로젝트 생성 또는 기존 프로젝트 선택
3. Vision API 활성화
4. API 키 생성 (API 및 서비스 → 사용자 인증 정보)

#### Qwen API (DashScope)
1. [Alibaba Cloud DashScope](https://dashscope.console.aliyun.com/) 접속
2. 계정 생성 및 로그인
3. API Key 발급

#### Replicate API
1. [Replicate](https://replicate.com/) 접속
2. 계정 생성 및 로그인
3. [API Tokens](https://replicate.com/account/api-tokens) 페이지에서 토큰 발급

### 3. 서버 실행

```bash
# 백엔드 서버 실행
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 프론트엔드 실행

프론트엔드는 정적 파일이므로 다음 중 하나의 방법으로 실행:

**방법 1: Python 간단 서버**
```bash
cd frontend
python -m http.server 3000
```

**방법 2: VS Code Live Server 확장 사용**

**방법 3: 직접 파일 열기**
- `frontend/index.html` 파일을 브라우저에서 직접 열기

### 5. 접속

- **프론트엔드**: http://localhost:3000
- **백엔드 API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

## 사용 방법

1. **사진 업로드**: 영상에 사용할 사진들을 업로드 (최대 10장)
2. **서사 입력**: 영상의 분위기와 스토리를 자연어로 설명
3. **스타일 선택**: 영상 스타일과 배경음악 분위기 선택
4. **영상 생성**: AI가 자동으로 1분짜리 쇼츠 영상 생성
5. **다운로드**: 완성된 영상 다운로드

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/generate` | 영상 생성 요청 |
| GET | `/api/progress/{task_id}` | 작업 진행 상황 조회 |
| DELETE | `/api/task/{task_id}` | 작업 취소 |
| GET | `/health` | 서버 상태 확인 |

## 개발 모드

API 키 없이 테스트하려면 프론트엔드의 데모 모드를 사용하세요.
`app.js`에서 `enableDemoMode()`가 자동으로 활성화되어 있어 백엔드 없이도 UI를 테스트할 수 있습니다.

## 라이선스

MIT License
