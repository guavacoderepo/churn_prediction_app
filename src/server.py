import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, status, Response
from typing import List

import pandas as pd
import mlflow
from contextlib import asynccontextmanager
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

from config.setting import Settings
from .services.modelingService import ModelingPipeline
from .services.etlService import ETLPipeline
from .utilities.redis_helpers import RedisHelper
from .schemas.responses import BatchResponse
from .schemas.predictions import Features, PredictionResult
from .utilities.mlflow_conn import mlflow_connect

settings = Settings()  # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async lifespan to initialize the app:
    - Load MLflow model
    - Connect to Redis
    """
    model_name = settings.MODEL_NAME
    alias = settings.ALIAS
    redis_url = f"redis://{settings.REDIS_URL}"

    r = await redis.Redis.from_url(redis_url, decode_responses=True)
    pong = await r.ping()
    print(f"Redis status - > {redis_url}:", "Connected" if pong else "Not connected")

    # Connect to MLflow / DagsHub
    exp_id = mlflow_connect()
    model_uri = f"models:/{model_name}_models@{alias}"
    model = mlflow.sklearn.load_model(model_uri)  # type: ignore

    # Store in app state
    app.state.model = model
    app.state.exp_id = exp_id
    app.state.redis = r

    # Update Prometheus metrics
    await update_metrics()

    print("✅ Model loaded successfully!")
    yield


# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API", version="1.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

# Prometheus metrics
accuracy_gauge = Gauge("mlflow_model_accuracy", "Accuracy per run", ["model_name"])
precision_gauge = Gauge("mlflow_model_precision", "Precision per run", ["model_name"])
f1_gauge = Gauge("mlflow_model_f1", "F1 Score per run", ["model_name"])
recall_gauge = Gauge("mlflow_model_recall", "Recall per run", ["model_name"])
model_gauge = Gauge("mlflow_model_count", "Total registered models")


async def update_metrics():
    """Fetch MLflow metrics and update Prometheus gauges"""
    experiment_id = app.state.exp_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    counter = 0

    try:
        for _, run in runs.iterrows():  # type: ignore
            if run.get("tags.model_name") == settings.MODEL_NAME:
                run_id = run.get("run_id", "Default")
                model_name = run.get("tags.model_name", 0)
                accuracy = run.get("metrics.accuracy", 0)
                precision = run.get("metrics.precision", 0)
                f1_score = run.get("metrics.f1", 0)
                recall = run.get("metrics.recall", 0)

                accuracy_gauge.labels(model_name=model_name).set(accuracy) 
                precision_gauge.labels(model_name=model_name).set(precision) 
                f1_gauge.labels(model_name=model_name).set(f1_score)
                recall_gauge.labels(model_name=model_name).set(recall)
                counter += 1

        model_gauge.set(counter)
    except:
        pass


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", status_code=status.HTTP_200_OK)
def home_route():
    """Welcome message for the API root"""
    return {
        "message": "🎉 Welcome to the Churn Prediction API! "
                   "Use /predict to get predictions and /train_model to retrain the model."
    }


@app.post("/predict", response_model=BatchResponse, status_code=status.HTTP_200_OK)
async def predict(features: List[Features]):
    """
    Perform churn prediction for a batch of features.
    Saves predictions to Redis and returns results.
    """
    try:
        feature_dicts = [f.model_dump(mode="json") for f in features]
        customer_ids = [f["customerID"] for f in feature_dicts]

        model = app.state.model
        if model is None:
            raise RuntimeError("Failed to load MLflow model.")

        pred_df = pd.DataFrame(feature_dicts)
        etl = ETLPipeline(source=pred_df)
        df_scaled, _ = etl.transform_data()

        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)
        preds_str = ["Yes" if p == 1 else "No" for p in preds]

        r = app.state.redis
        await RedisHelper(conn=r).save_data(feature_dicts, preds_str)

        results = [
            PredictionResult(
                customerID=cust_id,
                churn_score=round(prob[pred] * 100, 2),
                churn="Yes" if pred == 1 else "No"
            )
            for cust_id, prob, pred in zip(customer_ids, probs, preds)
        ]
        return BatchResponse(result=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting churn: {str(e)}")


@app.get("/train_model")
async def train_model():
    """
    Retrain the ML model using data from Redis + historical data.
    Updates MLflow metrics and clears Redis.
    """
    try:
        r = app.state.redis
        etl = ETLPipeline()

        # Retrieve new training data
        json_data = await RedisHelper(conn=r).retrieve_data()
        new_df = pd.DataFrame(json_data)
        old_df = etl.extract_data()

        # Combine datasets
        df = pd.concat([new_df, old_df], ignore_index=True)
        X_train, X_test, y_train, y_test = etl.tranform_split_data(df)
        y_train, y_test = y_train.values.ravel(), y_test.values.ravel()

        # Model training parameters
        params = {
            "max_depth": 17, "n_estimators": 285, "min_samples_leaf": 1,
            "min_samples_split": 2, "criterion": "gini"
        }

        # Train and log model
        modelling = ModelingPipeline((X_train, X_test, y_train, y_test))
        model = modelling.train_model(params=params)
        modelling.log_to_mlflow(model, settings.MODEL_NAME, params, settings.RUN_NAME)

        # Update Prometheus metrics
        await update_metrics()

        # Clear Redis
        await RedisHelper(conn=r).clear_data()

        return {"message": "✅ Model retraining and logging complete."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")
