from fastapi import FastAPI, HTTPException, status, Response
from .schemas.responses import BatchResponse
from .schemas.predictions import Features, PredictionResult
from prometheus_fastapi_instrumentator import Instrumentator
from .services.etl_pipeline import ETLPipeline
from typing import List
import pandas as pd
from config.setting import Settings
import mlflow 
from .utilities.mlflow_conn import mlflow_connect
from contextlib import asynccontextmanager


settings = Settings() # type: ignore
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = settings.MODEL_NAME
    alias = "champion"
    model_uri = f"models:/{model_name}_models@{alias}"

    # Connect to MLflow / DagsHub
    exp_id = mlflow_connect()
    model = mlflow.sklearn.load_model(model_uri) # type: ignore
    app.state.model = model  # store in app state
    app.state.exp_id = exp_id
    
    print("✅ Model loaded successfully!")
    yield  # important! signals app startup

app = FastAPI(title="churn prediction fastapi app", version="1.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # exposes /metrics

@app.get("/", status_code=status.HTTP_200_OK)
def home_route():
    return "hello world"

@app.post("/predict", response_model=BatchResponse, status_code=status.HTTP_200_OK)
def predict(features: List[Features]):
    try:
        # Convert Pydantic model to dict
        feature_dicts = [f.model_dump(mode="json") for f in features]
        # Extract customer IDs
        customer_ids = [f["customerID"] for f in feature_dicts]

        model = app.state.model  # fetch model from state
        if model is None:
            raise RuntimeError("Failed to load model from MLflow.")
        
        
        pred_df = pd.DataFrame(feature_dicts)
        etl = ETLPipeline(source=pred_df)
        df_scaled,_ = etl.transform_data()

        # Model inference
        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)

        # Build response list
        results = [
            PredictionResult(
                customerID=cust_id,
                churn_score=round(prob[pred]*100, 2),
                churn= "Yes" if pred == 1 else "No"
            )
            for cust_id, prob, pred in zip(customer_ids, probs, preds)
        ]
        return BatchResponse(result=results)

    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(status_code=500, detail=f"Error predicting churn: {str(e)}")

