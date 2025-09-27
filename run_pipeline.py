import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from src.services.etl_pipeline import ETLPipeline
from src.services.modeling_pipeline import ModelingPipeline
from src.utilities.mlflow_conn import mlflow_connect
from config.setting import Settings
from typing import Any
import joblib

def safe_int(value: Any) -> int | None:
    """Convert value safely to int, return None if not valid."""
    if value is None or value == "None":
        return None
    return int(value)

def load_models():
    """Fetch past MLflow runs and extract model parameters."""
    runs_info = []
    settings = Settings()  # type: ignore
    experiment_id = mlflow_connect()

    runs = mlflow.search_runs(experiment_ids=[experiment_id]) 
    for _, run in runs.iterrows():  # type: ignore
        if run.get('tags.model_name') == settings.MODEL_NAME:
            runs_info.append({
                "run_id": run.get('run_id'),
                "accuracy": run.get('metrics.accuracy'),
                "precision": run.get("metrics.precision"),
            })
    return runs_info

def etl_task():
    path = "/tmp/artifacts"
    file_name = "split_data.pkl"
    etl = ETLPipeline()
    X_train, X_test, y_train, y_test = etl.tranform_split_data()

    # save to temp files
    os.makedirs(path, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), f"{path}/{file_name}")
    return f"{path}/{file_name}"

def modelling_task(path):
    try:
        X_train, X_test, y_train, y_test = joblib.load(path)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        params = {'max_depth': 17, 'n_estimators': 285, 'min_samples_leaf': 1, 
                'min_samples_split': 2, 'criterion': 'gini'}

        modelling = ModelingPipeline((X_train, X_test, y_train, y_test))
        model = modelling.train_model(params=params)
        eval = modelling.evaluate_model(model=model)
        return model, params, eval
    except Exception as e:
        raise Exception(f"Error occured training model: {e}") from e

# def mflow_log():
#     """Execute ETL, train model, and log to MLflow."""
#     data = etl_pipeline()
#     sk_model, params, eval = modelling_task(data)
#     run_name = settings.RUN_NAME
#     model_name = settings.MODEL_NAME
#     # log_to_mlflow(sk_model, model_name, params, run_name)

data_path = etl_task()

model = modelling_task(data_path)
