# Quick Start Guide

## 빠른 시작 (5분)

### 1. 환경 설정
```bash
# 저장소 클론
git clone <repository-url>
cd AI

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env 파일 생성
cp .env-dev .env

# .env 파일 편집하여 Riot API Key 입력
RIOT_API_KEY=your_api_key_here
```

**Riot API Key 발급 방법**:
1. https://developer.riotgames.com/ 접속
2. Riot 계정으로 로그인
3. "DEVELOPMENT API KEY" 발급 (24시간 유효)

### 3. 서버 실행
```bash
# 개발 서버 실행
uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면 다음 URL에서 확인:
- API 문서: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## 기본 사용법

### API 테스트

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. 티어 목록 조회
```bash
curl http://localhost:8000/api/v1/tiers
```

#### 3. Gap 분석 테스트
```bash
curl -X POST http://localhost:8000/api/v1/analyze/gap \
  -H "Content-Type: application/json" \
  -d '{
    "player_stats": {
      "kills": 10,
      "deaths": 3,
      "assists": 8,
      "kda": 6.0,
      "cs": 180,
      "cs_per_min": 7.5,
      "gold": 15000,
      "vision_score": 35,
      "damage_dealt": 25000,
      "damage_share": 0.24,
      "champion_name": "Ahri"
    },
    "tier": "DIAMOND"
  }'
```

#### 4. 매치 분석 (실제 경기 ID 필요)
```bash
curl -X POST http://localhost:8000/api/v1/analyze/match \
  -H "Content-Type: application/json" \
  -d '{
    "match_id": "KR_7197963456",
    "summoner_name": "Hide on bush",
    "tag_line": "KR1"
  }'
```

---

## 프로젝트 구조 이해

```
AI/
├── src/                    # 소스 코드
│   ├── api/               # FastAPI 라우터
│   │   ├── routes.py      # API 엔드포인트
│   │   └── models.py      # Pydantic 모델
│   ├── models/            # AI 모델
│   │   └── rule_based.py  # Rule-based Gap 분석
│   ├── services/          # 비즈니스 로직
│   │   ├── riot_api.py    # Riot API 클라이언트
│   │   └── analyzer.py    # 경기 분석 서비스
│   └── config/            # 설정
│       └── settings.py    # 티어별 Baseline
├── scripts/               # 유틸리티 스크립트
│   └── collect_tier_baselines.py  # Baseline 데이터 수집
├── tests/                 # 테스트 코드
├── docs/                  # 문서
└── data/                  # 데이터 저장소
```

---

## 다음 단계

### Spring Boot 연동
`docs/SPRING_BOOT_INTEGRATION.md` 참고

### 티어별 Baseline 업데이트
```bash
python scripts/collect_tier_baselines.py
```

### 테스트 실행
```bash
pytest tests/ -v
```

### 배포
`docs/ARCHITECTURE.md` 참고

---

## 문제 해결

### Riot API Key 에러
```
401 Unauthorized
```
→ .env 파일의 API Key 확인

### Rate Limit 에러
```
429 Too Many Requests
```
→ Riot API 호출 제한 초과, 잠시 대기 후 재시도

### 모듈 Import 에러
```
ModuleNotFoundError: No module named 'src'
```
→ 가상환경 활성화 확인, requirements.txt 재설치

---

## 주요 문서

- **README.md**: 프로젝트 개요
- **ARCHITECTURE.md**: 시스템 아키텍처
- **SPRING_BOOT_INTEGRATION.md**: Spring Boot 연동
- **AUTHENTICATION.md**: 인증 전략

---

## 개발 팁

### 자동 재실행 (개발 중)
```bash
uvicorn src.api.routes:app --reload
```

### Swagger UI 활용
- http://localhost:8000/docs 에서 API 직접 테스트 가능
- Try it out 버튼으로 요청 보내기

### 로그 확인
```bash
tail -f logs/app.log
```

### 코드 포맷팅
```bash
black src/
```

### 타입 체크
```bash
mypy src/
```
