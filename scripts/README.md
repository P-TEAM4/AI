# Scripts 디렉토리

이 디렉토리에는 데이터 수집, 모델 학습 등의 유틸리티 스크립트가 포함되어 있습니다.

## 스크립트 목록

### 1. collect_tier_baselines.py
**목적**: Rule-based 모델의 티어별 Baseline 통계 수집

**사용법**:
```bash
python scripts/collect_tier_baselines.py
```

**출력**:
- `data/tier_baselines.json`: 수집된 티어별 평균 통계
- 콘솔에 Python 코드 출력 (settings.py에 복사하여 사용)

**소요 시간**: 약 1-2시간 (수집할 데이터 양에 따라 다름)

**주의사항**:
- Riot API Key 필요 (.env 파일에 설정)
- Rate Limit 준수 (자동으로 딜레이 적용)
- 네트워크 안정성 필요

---

## 데이터 수집 프로세스

### Step 1: 플레이어 리스트 수집
각 티어/디비전별로 랭크 플레이어 리스트를 가져옵니다.

```python
# IRON I 티어의 플레이어 30명
# BRONZE I 티어의 플레이어 30명
# SILVER I 티어의 플레이어 40명
# ...
```

### Step 2: 경기 데이터 수집
각 플레이어의 최근 5-10경기 데이터를 수집합니다.

### Step 3: 통계 계산
- KDA 평균
- CS/min 평균
- Gold 평균
- Vision Score 평균
- Damage Share 평균

### Step 4: 결과 저장
수집된 데이터를 JSON 파일로 저장하고, Python 코드를 생성합니다.

---

## Baseline 데이터 업데이트 방법

### 1. 데이터 수집
```bash
python scripts/collect_tier_baselines.py
```

### 2. 생성된 코드 복사
스크립트 실행 후 출력되는 Python 코드를 복사합니다.

### 3. settings.py 업데이트
`src/config/settings.py`의 `TierBaseline` 클래스에 붙여넣기:

```python
class TierBaseline:
    BASELINES: Dict[str, Dict[str, float]] = {
        # 여기에 복사한 코드 붙여넣기
    }
```

### 4. 서버 재시작
```bash
uvicorn src.api.routes:app --reload
```

---

## 추가 예정 스크립트

### train_ml_model.py
ML 모델 학습 스크립트 (Phase 2)

### evaluate_model.py
모델 성능 평가 스크립트

### update_baselines_auto.py
자동화된 Baseline 업데이트 (cron job)

### collect_training_data.py
ML 모델 학습용 대량 데이터 수집
