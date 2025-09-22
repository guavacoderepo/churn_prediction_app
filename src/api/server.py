from fastapi import FastAPI, HTTPException, status
from ..api.schemas.responses import BatchResponse
from ..api.schemas.request import BatchRequest
from ..api.schemas.predictions import Features, PredictionResult
from prometheus_fastapi_instrumentator import Instrumentator
from ..etl.transform import transform_data
from typing import List

from mlflow.sklearn import load_model

app = FastAPI(title="churn prediction fastapi app", version="1.0")

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

        model = load_model("models:/random_forest_models/2")
        if model is None:
            raise RuntimeError("Failed to load model from MLflow.")

        df_scaled = transform_data(feature_dicts)

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

