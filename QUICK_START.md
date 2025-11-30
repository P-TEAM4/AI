# Quick Start Guide

## 완료된 작업

1. **엑셀 데이터 전처리 완료** (40,410 → 20,556 records)
2. **티어별 베이스라인 학습 완료** (10개 티어, 실제 데이터 기반)
3. **동적 베이스라인 로더 구현** (학습된 데이터 자동 로드)
4. **main.py 생성** (Spring Boot 스타일 진입점)

---

## 학습된 데이터 통계

```
총 데이터: 20,556 경기
티어별 샘플 수:
- CHALLENGER: 223 경기
- GRANDMASTER: 1,121 경기
- MASTER: 3,332 경기
- DIAMOND: 5,421 경기
- EMERALD: 2,597 경기
- PLATINUM: 1,735 경기
- GOLD: 2,454 경기
- SILVER: 1,964 경기
- BRONZE: 1,249 경기
- IRON: 460 경기
```

---

## 서버 실행 (3단계)

### 1단계: 패키지 설치

```bash
cd AI
pip install -r requirements.txt
```

### 2단계: 환경 변수 설정 (선택)

`.env` 파일 생성 (Riot API 사용 시만 필요):
```bash
RIOT_API_KEY=your_api_key_here
```

### 3단계: 서버 실행

```bash
python main.py
```

**예상 출력:**
```
============================================================
LoL AI Analysis Service
============================================================
Environment: development
Host: 0.0.0.0
Port: 8000
Auto-reload: True
Log level: INFO
------------------------------------------------------------
API Docs: http://0.0.0.0:8000/docs
ReDoc: http://0.0.0.0:8000/redoc
Health Check: http://0.0.0.0:8000/health
============================================================
Using learned baselines from data/tier_baselines.json
   Trained at: 2025-11-28T...
   Total samples: 20556
RuleBasedGapAnalyzer initialized with learned baselines
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## API 테스트

### Health Check
```bash
curl http://localhost:8000/health
```

### Gap Analysis (예시)
```bash
curl -X POST http://localhost:8000/api/v1/analyze/gap \
  -H "Content-Type: application/json" \
  -d '{
    "player_stats": {
      "kills": 10,
      "deaths": 2,
      "assists": 15,
      "kda": 12.5,
      "cs": 200,
      "cs_per_min": 7.0,
      "gold": 15000,
      "vision_score": 35,
      "damage_dealt": 25000,
      "damage_share": 0.24,
      "champion_name": "Faker",
      "game_duration": 1800
    },
    "tier": "DIAMOND"
  }'
```

### Swagger UI
브라우저에서: `http://localhost:8000/docs`

---

## 학습된 파일 위치

```
AI/
├── data/
│   ├── processed_match_data.json     # 전처리된 데이터 (20,556 경기)
│   └── tier_baselines.json           # 학습된 베이스라인 
├── main.py                           # 서버 실행 파일 
├── src/
│   ├── config/
│   │   └── baseline_loader.py        # 동적 베이스라인 로더 
│   └── models/
│       └── rule_based.py             # 업데이트됨 
└── scripts/
    ├── preprocess_league_data.py     # 엑셀 → JSON 전처리 
    └── train_baselines.py            # 베이스라인 학습 
```

---

## 데이터 재학습 (향후)

### 새로운 엑셀 파일로 재학습

```bash
# 1. 전처리
python scripts/preprocess_league_data.py \
  --input /path/to/new_league_data.xlsx \
  --output data/processed_match_data.json

# 2. 학습
python scripts/train_baselines.py \
  --input data/processed_match_data.json \
  --display

# 3. 서버 재시작
python main.py
```

---

## 주요 특징

### 1. 실제 데이터 기반 베이스라인
- 하드코딩된 추정값 (이전)
- 20,556 경기 실제 데이터로 학습 (현재)

### 2. 자동 베이스라인 로딩
```python
# 서버 시작 시 자동으로:
# 1. data/tier_baselines.json 확인
# 2. 있으면 학습된 데이터 로드
# 3. 없으면 기본값 사용 (Fallback)
```

### 3. CS/min 이슈 해결 필요
현재 데이터에 CS 정보 없음 (0으로 표시)
→ CS 데이터 포함된 새 데이터셋 필요

---

## 트러블슈팅

### Q: cs_per_min이 0입니다
**A:** 현재 데이터셋에 CS 정보가 없습니다.
- 해결: `totalMinionsKilled`, `neutralMinionsKilled` 컬럼이 있는 데이터 필요
- 임시: CS를 제외한 다른 지표로 분석 가능

### Q: "No module named 'dotenv'" 에러
**A:** 패키지 설치:
```bash
pip install -r requirements.txt
```

### Q: 학습된 베이스라인이 적용되지 않음
**A:** 서버 재시작:
```bash
# Ctrl+C로 종료 후
python main.py
```

---

## 학습된 티어별 베이스라인

| Tier | KDA | Gold/min | Vision/min | Damage Share | Sample |
|------|-----|----------|------------|--------------|--------|
| **CHALLENGER** | 4.23 | 414.34 | 0.93 | 0.222 | 223 |
| **GRANDMASTER** | 4.13 | 403.74 | 0.87 | 0.207 | 1,121 |
| **MASTER** | 3.84 | 393.39 | 0.94 | 0.201 | 3,332 |
| **DIAMOND** | 3.51 | 391.21 | 0.91 | 0.203 | 5,421 |
| **EMERALD** | 3.49 | 391.42 | 0.87 | 0.211 | 2,597 |
| **PLATINUM** | 3.48 | 389.06 | 0.87 | 0.215 | 1,735 |
| **GOLD** | 3.23 | 386.61 | 0.83 | 0.212 | 2,454 |
| **SILVER** | 3.30 | 378.97 | 0.80 | 0.217 | 1,964 |
| **BRONZE** | 3.24 | 372.88 | 0.76 | 0.215 | 1,249 |
| **IRON** | 2.87 | 347.10 | 0.75 | 0.195 | 460 |

---

## 다음 단계

1. **서버 실행** → `python main.py`
2. **Spring Boot 연동 테스트**
3. **CS 데이터 포함된 새 데이터셋 확보** (추후)
4. **배포 후 Riot API 실시간 수집** (추후)

---

## 문의

학습 시스템 관련 문의:
- 학습 방법: `README_BASELINE_TRAINING.md` 참고
- API 문서: `http://localhost:8000/docs`
- 아키텍처: `docs/ARCHITECTURE.md`
