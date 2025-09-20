# Wine Quality Prediction with PySpark & MLflow

이 프로젝트는 PySpark와 MLflow를 사용하여 와인 품질 예측 모델을 학습하고 관리하는 머신러닝 프로젝트입니다.

## 📁 프로젝트 구조

```
wine/
├── wine_pyspark.py              # 메인 학습 스크립트
├── wine_quality_merged.csv      # 와인 품질 데이터셋
├── docker-compose.yml           # Docker Compose 설정
├── Dockerfile.mlflow            # MLflow UI용 Dockerfile
├── Dockerfile.training          # 학습용 Dockerfile
├── requirements.txt             # Python 의존성
├── run.sh                       # 로컬 실행 스크립트
├── mlruns/                      # MLflow 실험 결과 저장소
└── README.md                    # 프로젝트 문서
```

## 🚀 빠른 시작

### Docker Compose 사용 (권장)

1. **모든 서비스 실행 (MLflow UI + 학습)**
   ```bash
   docker-compose up -d
   ```

2. **MLflow UI만 실행**
   ```bash
   docker-compose up -d mlflow-ui
   ```

3. **학습만 실행**
   ```bash
   docker-compose up training
   ```

4. **MLflow UI 접속**
   - 브라우저에서 http://localhost:5000 접속
   - 실험 결과, 모델, 메트릭 확인 가능

### 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 학습 실행
python wine_pyspark.py

# MLflow UI 실행
mlflow ui --host 0.0.0.0 --port 5000
```

## 🧠 모델 정보

### 사용된 알고리즘
- **Neural Network (MultilayerPerceptronClassifier)**
- **Naive Bayes**
- **Logistic Regression**

### 데이터 전처리
- 라벨 재분류: 품질 3-5 → 0 (낮음), 6-9 → 1 (높음)
- 특성 벡터화 및 정규화
- 교차 검증을 통한 하이퍼파라미터 튜닝

### 평가 메트릭
- **Accuracy**: 정확도
- **F1-score**: F1 점수

## 📊 실험 관리

### MLflow 기능
- **실험 추적**: 각 실행의 파라미터, 메트릭, 아티팩트 저장
- **모델 버전 관리**: 최적 모델 자동 등록
- **비교 분석**: 여러 모델 성능 비교
- **재현성**: 실험 환경 및 코드 버전 관리

### 실험 결과 확인
1. http://localhost:5000 접속
2. "WineQualityExperiment" 실험 선택
3. 각 실행의 상세 정보 확인
4. 모델 성능 비교 및 최적 모델 선택

## 🐳 Docker 서비스 구성

### mlflow-ui
- **포트**: 5000
- **기능**: MLflow 웹 인터페이스
- **볼륨**: `./mlruns:/app/mlruns`

### training
- **기능**: 모델 학습 및 실험 실행
- **볼륨**: 
  - `./mlruns:/app/mlruns` (실험 결과 저장)
  - `./wine_quality_merged.csv:/app/wine_quality_merged.csv` (데이터)

## 📈 실험 결과 예시

```
Best Params: {'maxIter': 50, 'blockSize': 64}
Accuracy: 0.6436041834271923
F1-score: 0.5040463502121917
```

## 🔧 개발 환경

- **Python**: 3.9+
- **PySpark**: 3.5.0
- **MLflow**: 2.7.0
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

## 📝 주요 기능

1. **자동화된 실험**: 여러 알고리즘 자동 학습 및 비교
2. **하이퍼파라미터 튜닝**: 교차 검증을 통한 최적 파라미터 탐색
3. **모델 버전 관리**: 최적 모델 자동 등록 및 관리
4. **실시간 모니터링**: MLflow UI를 통한 실험 진행 상황 확인
5. **재현 가능한 실험**: Docker를 통한 일관된 환경 제공

## 🚨 문제 해결

### Docker 관련 오류
- Docker Desktop이 실행 중인지 확인
- `docker-compose down` 후 `docker-compose up -d --build` 재실행

### MLflow UI 접속 불가
- 포트 5000이 사용 중인지 확인
- 방화벽 설정 확인

### 데이터베이스 오류
- 초기 실행 시 정상적인 현상
- 실험 실행 후 자동으로 해결됨

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. Docker Desktop 실행 상태
2. 포트 5000 사용 가능 여부
3. `mlruns` 폴더 권한 설정