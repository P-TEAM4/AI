# LoL 자동 하이라이트 생성 및 분석 AI 시스템

## 프로젝트 개요
League of Legends (LoL) 경기 데이터를 분석하여 자동으로 하이라이트를 생성하고 승패 요인을 분석하는 AI 시스템입니다.

## 주요 기능
- **Rule-based 모델**: 플레이어 스탯과 티어 평균을 비교하여 갭(Gap) 분석
  - 게임 시간 정규화: 골드/분, 비전 스코어/분으로 공정한 비교
  - 5가지 지표 분석: KDA, CS/min, Gold/min, Vision/min, Damage Share
- **Riot API 연동**: Match, Timeline, Rank Tier 정보 수집
- **Spring Boot 연동**: REST API를 통한 백엔드 서버 연동
- **분석 리포트 생성**: 강점/약점 식별, 개선 추천사항 제공
- **시각화 지원**: 테스트/개발용 성능 차트 생성 (Radar, Bar, Overview)

## 기술 스택
- **언어**: Python 3.11+
- **프레임워크**: FastAPI
- **API**: Riot Games API
- **시각화**: Matplotlib, Seaborn (개발/테스트용)
- **테스트**: pytest, pytest-cov
- **데이터베이스**: TBD (PostgreSQL/Redis 고려 중)

## 프로젝트 구조
```
AI/
├── src/
│   ├── api/                 # FastAPI 라우터 및 엔드포인트
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── models.py       # Pydantic 모델
│   ├── models/             # AI 모델
│   │   ├── __init__.py
│   │   └── rule_based.py   # Rule-based Gap 계산 모델
│   ├── services/           # 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── riot_api.py     # Riot API 클라이언트
│   │   └── analyzer.py     # 경기 분석 서비스
│   ├── utils/              # 유틸리티 함수
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   └── visualizer.py   # 시각화 유틸리티 (테스트용)
│   └── config/             # 설정 파일
│       ├── __init__.py
│       └── settings.py     # 티어 베이스라인 포함
├── scripts/                # 유틸리티 스크립트
│   ├── collect_tier_baselines.py  # 티어 데이터 수집
│   ├── visualize_example.py       # 시각화 예제
│   └── test_real_api.py           # 실제 API 테스트
├── tests/                  # 테스트 코드
│   ├── test_api.py         # API 엔드포인트 테스트
│   ├── test_riot_api.py    # Riot API 클라이언트 테스트
│   ├── test_analyzer.py    # 분석 서비스 테스트
│   ├── test_rule_based.py  # 룰 기반 모델 테스트
│   └── test_evaluation.py  # 모델 평가 테스트
├── visualization_results/  # 시각화 결과 저장 (gitignore)
├── data/                   # 데이터 저장소
├── logs/                   # 로그 파일
├── requirements.txt        # Python 패키지 의존성
├── pytest.ini             # pytest 설정
├── .env-dev               # 환경 변수 (gitignore)
├── .gitignore
└── README.md
```

## 설치 및 실행

### 1. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정
```bash
cp .env-dev .env
# .env 파일에 Riot API Key 설정
```

### 4. 서버 실행
```bash
uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

## API 엔드포인트

### 1. 경기 분석
```
POST /api/v1/analyze/match
Content-Type: application/json

{
  "match_id": "KR_1234567890",
  "summoner_name": "Hide on bush",
  "tag_line": "KR1"
}
```

### 2. 플레이어 프로파일 분석
```
POST /api/v1/analyze/profile
Content-Type: application/json

{
  "summoner_name": "Hide on bush",
  "tag_line": "KR1",
  "recent_games": 20
}
```

### 3. Gap 분석
```
POST /api/v1/analyze/gap
Content-Type: application/json

{
  "player_stats": {
    "kda": 3.5,
    "cs_per_min": 8.2,
    "gold": 15000,
    "vision_score": 45
  },
  "tier": "DIAMOND",
  "division": "II"
}
```

## Rule-based 모델 설명

### Gap 계산 로직
1. **티어별 표준 모델(Baseline)**: 각 티어의 평균 지표(KDA, CS/min, Gold 등)를 정의
2. **개인 프로파일링**: 플레이어의 최근 경기 데이터로 개인 평균 계산
3. **Gap 분석**:
   - 개인 프로필과 티어별 표준 모델 비교
   - 정규화된 차이값 계산
   - 승리/패배 기여도 산출

### 티어별 Baseline 데이터 수집
Rule-based 모델의 정확도를 높이기 위해 실제 데이터를 수집하여 Baseline을 업데이트할 수 있습니다.

```bash
# 티어별 Baseline 데이터 수집 (약 1-2시간 소요)
python scripts/collect_tier_baselines.py

# 수집된 데이터는 data/tier_baselines.json에 저장됨
# 해당 데이터를 src/config/settings.py의 TierBaseline 클래스에 반영
```

**주의사항**:
- Riot API Rate Limit 준수 필요 (개발용 API Key: 20 req/sec)
- 대량 데이터 수집 시 시간이 오래 걸릴 수 있음
- 정기적으로 업데이트하여 메타 변화 반영 (월 1회 권장)

### 분석 지표 (게임 시간 정규화 적용)
- **KDA** (Kill/Death/Assist Ratio) - 게임별 독립 지표
- **CS/min** (Creep Score per Minute) - 분당 CS
- **Gold/min** (Gold per Minute) - ⚠️ **분당 골드로 정규화** (게임 길이 보정)
- **Vision/min** (Vision Score per Minute) - ⚠️ **분당 비전 스코어로 정규화** (게임 길이 보정)
- **Damage Share** - 팀 내 딜 비중 (%)

**정규화 이유**:
- 짧은 게임(20분)과 긴 게임(40분)을 공정하게 비교하기 위함
- 절대값 대신 분당 지표 사용으로 게임 길이 영향 제거

## Git 브랜치 전략
- `main`: 프로덕션 배포 브랜치 (deploy)
- `develop`: 개발 통합 브랜치
- `feature/*`: 새로운 기능 개발 브랜치
- `fix/*`: 버그 수정 브랜치
- `refactor/*`: 코드 리팩토링 브랜치
- `hotfix/*`: 긴급 수정 브랜치
- `test/*`: 테스트 코드 작성 브랜치
- `docs/*`: 문서 작업 브랜치

### 브랜치 작업 흐름
1. `feature/*`, `fix/*`, `refactor/*` 등의 작업 브랜치에서 개발
2. 작업 완료 후 `develop` 브랜치로 Pull Request
3. 코드 리뷰 후 `develop`에 머지
4. `develop`에서 테스트 및 검증 완료 후 `main`으로 머지
5. `main` 브랜치는 자동으로 프로덕션 배포

## 테스트

### 테스트 실행
```bash
# 모든 테스트 실행
pytest

# 특정 카테고리만 실행
pytest -m unit          # 단위 테스트
pytest -m integration   # 통합 테스트
pytest -m evaluation    # 모델 평가 테스트

# 커버리지 리포트 생성
pytest --cov=src --cov-report=html
```

### 실제 API 테스트
```bash
# Riot API로 실제 데이터 가져와서 테스트
python scripts/test_real_api.py

# 결과: 계정 조회, 매치 분석, Gap 분석, 시각화 생성
```

## 시각화 (개발/테스트용)

### 시각화 예제 실행
```bash
# 테스트 데이터로 시각화 생성
python scripts/visualize_example.py

# 결과: visualization_results/ 디렉토리에 PNG 파일 생성
```

### 생성되는 차트
1. **Performance Overview**: 종합 대시보드 (4개 차트 통합)
2. **Radar Chart**: 5가지 지표 레이더 차트
3. **Gap Bars**: 티어 평균 대비 갭 막대 그래프
4. **Tier Comparison**: 모든 티어와의 매칭도 비교

**주의**: 시각화는 테스트/개발 목적으로만 사용됩니다. API 엔드포인트로는 제공되지 않습니다.

## 베이스라인 데이터

현재 `src/config/settings.py`의 티어별 베이스라인 데이터는:
- **임시 추정값**: 롤 통계 사이트 기반 추정
- **실제 수집 필요**: `scripts/collect_tier_baselines.py` 실행하여 실제 데이터 수집 권장
- **정규화 적용**: Gold/min, Vision/min으로 게임 시간 보정

## 개발 팀
- **AI 개발**: 송재곤, 김문기

## 라이선스
TBD
