import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Spark 세션
spark = SparkSession.builder.appName("WineNN_Tuning").getOrCreate()

# CSV 데이터 불러오기
data = spark.read.csv("wine_quality_merged.csv", header=True, inferSchema=True)

# 실험 이름 설정
mlflow.set_experiment("WineQualityExperiment")

# 라벨 재분류: 3~5=0, 6~9=1
data = data.withColumn("label_class", when(data.quality <= 5, 0).otherwise(1))

# type 컬럼 one-hot encoding
if "type" in data.columns:
    type_indexer = StringIndexer(inputCol="type", outputCol="type_index")
    data = type_indexer.fit(data).transform(data)
    encoder = OneHotEncoder(inputCols=["type_index"], outputCols=["type_vec"])
    data = encoder.fit(data).transform(data)
    feature_cols = [c for c in data.columns if c not in ("quality", "type", "label_class")] + ["type_vec"]
else:
    feature_cols = [c for c in data.columns if c not in ("quality", "label_class")]

# 특징 벡터화
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
data = assembler.transform(data)

# 수치형 스케일링
scaler = StandardScaler(inputCol="features_raw", outputCol="features")
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

# 학습/테스트 분할
train_ratio = 0.8
test_ratio = 0.2
train, test = data.randomSplit([train_ratio, test_ratio], seed=42)

# MLflow 실험 실행
with mlflow.start_run(run_name="NN_Wine_Quality_Tuning"):
    mlflow.log_param("num_features", len(feature_cols))
    mlflow.log_param("train_ratio", train_ratio)
    mlflow.log_param("test_ratio", test_ratio)
    mlflow.log_param("model_type", "MultilayerPerceptronClassifier")

    # 🔹 NN 모델 정의
    # layers: [입력층, 은닉층1, 은닉층2, 출력층]
    layers = [len(feature_cols), 10, 5, 2]  # 출력 2개 = 이진 분류
    nn = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label_class",
                                        maxIter=100, layers=layers, blockSize=128, seed=42)

    # 🔹 하이퍼파라미터 그리드 (maxIter, blockSize 등)
    paramGrid = (ParamGridBuilder()
                 .addGrid(nn.maxIter, [50, 100])
                 .addGrid(nn.blockSize, [64, 128])
                 .build())

    # 🔹 Cross-validation
    evaluator = MulticlassClassificationEvaluator(labelCol="label_class", predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=nn, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

    # 학습
    cv_model = cv.fit(train)

    # 베스트 모델 선택
    best_model = cv_model.bestModel
    best_params = {
        "maxIter": best_model.getOrDefault("maxIter"),
        "blockSize": best_model.getOrDefault("blockSize")
    }
    mlflow.log_params(best_params)

    # 테스트 데이터 예측
    predictions = best_model.transform(test)

    # 평가
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label_class", predictionCol="prediction", metricName="f1")
    f1 = f1_evaluator.evaluate(predictions)

    print("Best Params:", best_params)
    print("Accuracy:", accuracy)
    print("F1-score:", f1)

    # MLflow에 기록
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # 모델 저장
    mlflow.spark.log_model(best_model, "nn_model")
