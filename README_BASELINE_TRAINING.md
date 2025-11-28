# 베이스라인 학습 시스템 가이드

## 개요

이 시스템은 **학습된 데이터로 티어별 기준값(Baseline)을 자동으로 생성**합니다.

### 구조
```
하드코딩 베이스라인 (settings.py)
         ↓
학습된 베이스라인 (data/tier_baselines.json) ← 우선 사용
         ↓
API 실시간 업데이트 (배포 후 구현 예정)
```

---

## Phase 1: 로컬 데이터로 학습 (현재)

### 1단계: 데이터 준비

다운로드한 매치 데이터를 `data/` 폴더에 저장합니다.

**필수 컬럼:**
- `tier`: 플레이어 티어 (GOLD, PLATINUM, DIAMOND 등)
- `kda`: KDA 비율
- `game_duration`: 게임 시간 (초)
- `gold`: 획득 골드
- `vision_score`: 비전 스코어
- `damage_share`: 팀 내 딜 비중
- `cs_per_min` (선택): CS/분 (자동 계산 가능)

**지원 포맷:**
- JSON (`.json`)
- CSV (`.csv`)
- Parquet (`.parquet`)

**예시 데이터:**
```json
[
  {
    "tier": "GOLD",
    "kda": 3.2,
    "cs_per_min": 6.1,
    "gold": 13500,
    "vision_score": 28,
    "damage_share": 0.22,
    "game_duration": 1800,
    "win": true
  }
]
```

### 2단계: 베이스라인 학습

```bash
# 기본 사용법
python scripts/train_baselines.py --input data/your_match_data.json

# 결과 확인하면서 학습
python scripts/train_baselines.py --input data/your_match_data.json --display

# 커스텀 출력 경로
python scripts/train_baselines.py --input data/your_match_data.json --output data/custom_baseline.json
```

### 3단계: 서버 실행

학습된 베이스라인이 자동으로 로드됩니다.

```bash
python main.py
```

**출력 예시:**
```
Using learned baselines from data/tier_baselines.json
   Trained at: 2025-11-28T10:30:00
   Total samples: 1234
RuleBasedGapAnalyzer initialized with learned baselines
```

---

## 예제: 테스트 데이터로 학습

```bash
# 제공된 예제 데이터로 테스트
python scripts/train_baselines.py --input data/example_match_data.json --display
```

**예상 출력:**
```
====================================================================================================
LEARNED TIER BASELINES
====================================================================================================
          avg_kda  avg_cs_per_min  avg_gold_per_min  avg_vision_score_per_min  avg_damage_share
GOLD         2.85            5.95            456.82                      0.87              0.205
PLATINUM     3.45            6.55            487.50                      1.02              0.225
DIAMOND      4.20            7.80            533.33                      1.22              0.260
====================================================================================================
```

---

## Phase 2: API로 실시간 수집 (배포 후)

### Riot API로 데이터 수집

```bash
# 티어별 실제 데이터 수집 (약 1-2시간 소요)
python scripts/collect_tier_baselines.py
```

**주의사항:**
- Riot API Rate Limit 준수 필요
- 개발용 API Key: 20 req/sec, 100 req/2min
- 프로덕션 API Key 권장

**수집 후 자동 학습:**
```bash
# collect_tier_baselines.py가 data/tier_baselines.json 생성
# 서버 재시작 시 자동 적용
python main.py
```

---

## 🛠 코드 구조

### 1. `baseline_trainer.py` (학습 엔진)
```python
from src.services.baseline_trainer import BaselineTrainer

trainer = BaselineTrainer()

# 데이터 로드
df = trainer.load_match_data("data/matches.json")

# 학습
baselines = trainer.train_from_data(df)

# 저장
trainer.save_baselines()
```

### 2. `baseline_loader.py` (동적 로더)
```python
from src.config.baseline_loader import get_baseline_loader

loader = get_baseline_loader()

# 자동으로 학습된 파일 또는 기본값 로드
baseline = loader.get_baseline("GOLD")

# 학습된 베이스라인 사용 여부 확인
if loader.is_using_learned_baselines():
    print("Using learned baselines")
```

### 3. `rule_based.py` (분석 모델)
```python
from src.models.rule_based import RuleBasedGapAnalyzer

analyzer = RuleBasedGapAnalyzer()
# 자동으로 baseline_loader 사용
# 학습된 데이터가 있으면 우선 사용
```

---

## 파일 구조

```
AI/
├── data/
│   ├── example_match_data.json        # 예제 데이터
│   └── tier_baselines.json            # 학습된 베이스라인 (생성됨)
├── src/
│   ├── config/
│   │   ├── settings.py                # 기본 베이스라인 (Fallback)
│   │   └── baseline_loader.py         # 동적 로더
│   ├── services/
│   │   └── baseline_trainer.py        # 학습 엔진
│   └── models/
│       └── rule_based.py              # 분석 모델 (업데이트됨)
└── scripts/
    ├── train_baselines.py             # 로컬 데이터 학습
    └── collect_tier_baselines.py      # Riot API 수집 (기존)
```

---

## 베이스라인 우선순위

1. **학습된 베이스라인** (`data/tier_baselines.json`)
   - 실제 데이터 기반
   - 가장 정확함
   - **우선 사용**

2. **기본 베이스라인** (`src/config/settings.py`)
   - 하드코딩된 추정값
   - Fallback으로만 사용
   - 학습 파일 없을 때만 사용

---

## 🔧 유지보수

### 베이스라인 업데이트 주기

```bash
# 메타 변화 후 재학습 (월 1회 권장)
python scripts/collect_tier_baselines.py
# 자동으로 data/tier_baselines.json 업데이트
# 서버 재시작
python main.py
```

### 베이스라인 재로드 (서버 재시작 없이)

```python
from src.config.baseline_loader import get_baseline_loader

loader = get_baseline_loader()
loader.reload_baselines()  # 파일에서 다시 로드
```

---

## 트러블슈팅

### Q: "No baseline file found" 메시지가 나옵니다
**A:** 정상입니다. 기본 베이스라인을 사용합니다. 학습하려면:
```bash
python scripts/train_baselines.py --input data/your_data.json
```

### Q: 학습 후에도 기본 베이스라인이 사용됩니다
**A:** 서버를 재시작하세요:
```bash
# Ctrl+C로 종료 후
python main.py
```

### Q: 데이터 컬럼이 맞지 않습니다
**A:** 필수 컬럼 확인:
```python
required = ["tier", "kda", "game_duration", "gold", "vision_score", "damage_share"]
```

---

## 다음 단계

1. **로컬 데이터 학습** (현재)
2. **API 자동 수집** (collect_tier_baselines.py)
3. **자동 업데이트 스케줄러** (배포 후)
4. **베이스라인 버전 관리** (향후)

---

## 팁

- 샘플 데이터가 많을수록 정확도 향상 (티어당 최소 100경기 권장)
- 게임 시간 정규화가 자동으로 수행됨 (gold_per_min, vision_score_per_min)
- 학습된 베이스라인은 Git에 커밋하지 않음 (.gitignore 적용)
- 프로덕션 환경에서는 주기적으로 재학습 권장
