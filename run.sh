#!/bin/bash

# MLflow 아티팩트 경로를 /app/mlruns로 지정
MLRUNS_DIR=/app/mlruns

# 모델 학습 + MLflow 기록
python3 wine_pyspark.py

# MLflow UI 실행
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri $MLRUNS_DIR
# --default-artifact-root $MLRUNS_DIR