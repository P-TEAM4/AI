# AI 아키텍처 설계 문서

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      Desktop Frontend                            │
│                     (Electron + React)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────┴──────────────────────────────────────┐
│                    Spring Boot Backend                           │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  Controller  │    │   Service    │    │  Repository  │     │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│          │                   │                   │              │
└──────────┼───────────────────┼───────────────────┼──────────────┘
           │                   │                   │
           │ REST API          │                   │
           ├──────────┬────────┘                   │
           │          │                            │
┌──────────┴─────┐    │                   ┌────────┴─────────┐
│   AI Service   │    │                   │    Database      │
│   (FastAPI)    │    │                   │  (PostgreSQL)    │
│                │    │                   └──────────────────┘
│  ┌──────────┐  │    │
│  │Rule-based│  │    │
│  │  Model   │  │    │
│  └──────────┘  │    │
│                │    │
│  ┌──────────┐  │    │
│  │ ML Model │  │    │
│  └──────────┘  │    │
└────────┬───────┘    │
         │            │
         ├────────────┘
         │ HTTP/REST
┌────────┴──────────────────┐         ┌──────────────────┐
│    Riot Games API         │         │   Redis Cache    │
│ (Match, Timeline, Rank)   │◄────────┤   (선택사항)      │
└───────────────────────────┘         └──────────────────┘
```

## 주요 컴포넌트

### 1. FastAPI AI Service
**책임**:
- Rule-based 모델을 사용한 Gap 분석
- Riot API 데이터 수집 및 처리
- 플레이어 성능 분석
- 하이라이트 추출 (향후)

**주요 엔드포인트**:
- `POST /api/v1/analyze/match`: 경기 분석
- `POST /api/v1/analyze/profile`: 프로파일 분석
- `POST /api/v1/analyze/gap`: Gap 분석

### 2. Spring Boot Backend
**책임**:
- 사용자 인증 및 권한 관리
- 비즈니스 로직 처리
- AI 서비스 호출 및 결과 저장
- 프론트엔드 API 제공

### 3. Database (PostgreSQL)
**스키마 설계 예시**:
```sql
-- 사용자 테이블
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 플레이어 정보 테이블
CREATE TABLE players (
    id SERIAL PRIMARY KEY,
    puuid VARCHAR(78) UNIQUE NOT NULL,
    summoner_name VARCHAR(50) NOT NULL,
    tag_line VARCHAR(10) NOT NULL,
    tier VARCHAR(20),
    division VARCHAR(5),
    lp INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 매치 분석 결과 테이블
CREATE TABLE match_analyses (
    id SERIAL PRIMARY KEY,
    match_id VARCHAR(50) UNIQUE NOT NULL,
    player_id INTEGER REFERENCES players(id),
    analysis_result JSONB NOT NULL,
    overall_score DECIMAL(5,2),
    win BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 프로파일 분석 결과 테이블
CREATE TABLE profile_analyses (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    games_analyzed INTEGER,
    avg_stats JSONB,
    gap_analysis JSONB,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스
CREATE INDEX idx_match_id ON match_analyses(match_id);
CREATE INDEX idx_player_puuid ON players(puuid);
CREATE INDEX idx_analyzed_at ON match_analyses(created_at DESC);
```

### 4. Redis Cache (선택사항)
**캐싱 전략**:
```python
# 매치 데이터: 24시간 캐싱
CACHE_TTL_MATCH = 86400

# 플레이어 랭크: 1시간 캐싱
CACHE_TTL_RANK = 3600

# Gap 분석 결과: 30분 캐싱
CACHE_TTL_ANALYSIS = 1800
```

## 데이터 플로우

### 경기 분석 플로우
```
1. User → Frontend: 경기 분석 요청
2. Frontend → Spring Boot: POST /api/matches/analyze
3. Spring Boot → AI Service: POST /api/v1/analyze/match
4. AI Service → Riot API: Match 데이터 조회
5. AI Service: Rule-based 모델로 분석
6. AI Service → Spring Boot: 분석 결과 반환
7. Spring Boot → Database: 결과 저장
8. Spring Boot → Frontend: 분석 결과 반환
9. Frontend → User: 결과 표시
```

### 캐싱 플로우 (Redis 사용 시)
```
1. AI Service: Match ID로 캐시 조회
2. Cache Hit → 캐시 데이터 반환
3. Cache Miss → Riot API 호출
4. AI Service: 데이터를 캐시에 저장 (TTL 설정)
5. AI Service: 분석 수행
```

## 확장성 고려사항

### 수평 확장 (Horizontal Scaling)
```yaml
# Docker Compose 예시
version: '3.8'
services:
  ai-service:
    image: lol-ai-service:latest
    deploy:
      replicas: 3  # 3개의 인스턴스 실행
    environment:
      - RIOT_API_KEY=${RIOT_API_KEY}

  nginx:
    image: nginx:latest
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - ai-service
```

### 비동기 작업 처리 (Celery)
```python
# tasks.py
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def analyze_profile_async(summoner_name, tag_line, num_games):
    """비동기 프로파일 분석"""
    analyzer = MatchAnalyzer()
    result = analyzer.analyze_profile(summoner_name, tag_line, num_games)
    return result

# 사용
task = analyze_profile_async.delay("Player", "KR1", 20)
result = task.get(timeout=30)
```

## 보안 아키텍처

### API Key 관리
```
1. GitHub Secrets: CI/CD 파이프라인에서 사용
2. 환경 변수: 로컬 개발 및 서버 배포
3. AWS Secrets Manager: 프로덕션 환경 (선택)
4. HashiCorp Vault: 엔터프라이즈 환경 (선택)
```

### 인증/인가
```
┌──────────┐      JWT Token      ┌──────────────┐
│  Client  │ ──────────────────► │ Spring Boot  │
└──────────┘                      │   (Gateway)  │
                                  └──────┬───────┘
                                         │
                                  Token Validation
                                         │
                                  ┌──────┴───────┐
                                  │ AI Service   │
                                  │ (Protected)  │
                                  └──────────────┘
```

## 모니터링 및 로깅

### Logging 구조
```python
# 로그 레벨
DEBUG: 개발 환경 상세 정보
INFO: 일반 작업 로그
WARNING: 경고 (Rate Limit 근접 등)
ERROR: 에러 발생
CRITICAL: 심각한 오류

# 로그 포맷
{
  "timestamp": "2025-11-27T10:00:00Z",
  "level": "INFO",
  "service": "ai-service",
  "endpoint": "/api/v1/analyze/match",
  "match_id": "KR_1234567890",
  "duration_ms": 1250,
  "status": "success"
}
```

### 메트릭 수집
```python
# Prometheus 메트릭 예시
api_requests_total: 총 API 요청 수
api_request_duration_seconds: API 응답 시간
riot_api_calls_total: Riot API 호출 수
cache_hit_rate: 캐시 히트율
model_inference_time: 모델 추론 시간
```

## 성능 목표

| 메트릭 | 목표 | 현재 |
|--------|------|------|
| API 응답 시간 (95th percentile) | < 2초 | TBD |
| 처리량 (Throughput) | 100 req/sec | TBD |
| 가용성 (Availability) | 99.9% | TBD |
| 에러율 (Error Rate) | < 0.1% | TBD |

## 배포 전략

### Blue-Green Deployment
```
1. Blue (현재 버전) 운영 중
2. Green (새 버전) 배포 및 테스트
3. 트래픽을 Green으로 전환
4. Blue 모니터링 후 제거
```

### Canary Deployment
```
1. 새 버전을 10%의 트래픽으로 배포
2. 모니터링 및 검증
3. 점진적으로 트래픽 증가 (10% → 50% → 100%)
4. 문제 발생 시 즉시 롤백
```

## 재해 복구 (Disaster Recovery)

### 백업 전략
```
- 데이터베이스: 일일 자동 백업
- 모델 파일: 버전별 저장
- 설정 파일: Git 버전 관리
- 로그: 7일간 보관
```

### 장애 시나리오
```
시나리오 1: Riot API 장애
→ 캐시된 데이터 사용 (Fallback)
→ 사용자에게 안내 메시지

시나리오 2: AI 서비스 장애
→ Spring Boot에서 Health Check 실패 감지
→ 자동 재시작 (Docker/K8s)
→ 알림 발송

시나리오 3: 데이터베이스 장애
→ Read Replica 사용 (읽기 전용)
→ 쓰기 작업 큐잉
→ Primary DB 복구 후 재처리
```

## 확장 계획

### Phase 1: MVP (현재)
- Rule-based 모델
- 기본 Gap 분석
- Spring Boot 연동

### Phase 2: ML 모델 도입
- XGBoost/RandomForest 모델 학습
- Feature Engineering
- 모델 A/B 테스트

### Phase 3: 고급 기능
- 실시간 스트리밍 분석
- 하이라이트 영상 자동 생성
- 개인화된 코칭 추천
- 챔피언 메타 분석

### Phase 4: 스케일 아웃
- Kubernetes 클러스터
- 멀티 리전 배포
- CDN 적용
- 마이크로서비스 분리
