import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from airflow import DAG
# from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.python import PythonOperator # type: ignore
from datetime import timedelta, datetime
from src.services.etl_pipeline import ETLPipeline
from src.services.modeling_pipeline import ModelingPipeline
from utilities.mlflow_conn import mlflow_connect
from config.setting import Settings
from typing import Any
import joblib

# Default DAG arguments
default_args = {
    "owner": "churn_pred_app",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
    "start_date": datetime.now()
}

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
                "params": {
                    "n_estimators": safe_int(run.get("params.n_estimators")),
                    "criterion": run.get("params.criterion"),
                    "min_samples_leaf": safe_int(run.get("params.min_samples_leaf")),
                    "min_samples_split": safe_int(run.get("params.min_samples_split")),
                    "max_depth": safe_int(run.get("params.max_depth")),
                }
            })
    return runs_info

def etl_task():
    path = "/tmp/airflow_data"
    file_name = "split_data.pkl"
    etl = ETLPipeline()
    X_train, X_test, y_train, y_test = etl.tranform_split_data()

    # save to temp files
    os.makedirs(path, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), f"{path}/{file_name}")
    # push path to XCom
    return f"{path}/{file_name}"

def modelling_task(**context):
    path = context['ti'].xcom_pull(task_ids='etl_task')
    X_train, X_test, y_train, y_test = joblib.load(path)
    params = load_models()[0]['params']

    modelling = ModelingPipeline((X_train, X_test, y_train, y_test))
    model = modelling.train_model(params=params)
    eval = modelling.evaluate_model(model=model)
    return model, params, eval

# def mflow_log():
#     """Execute ETL, train model, and log to MLflow."""
#     data = etl_pipeline()
#     sk_model, params, eval = modelling_task(data)
#     run_name = settings.RUN_NAME
#     model_name = settings.MODEL_NAME
#     # log_to_mlflow(sk_model, model_name, params, run_name)


# Airflow DAG definition
with DAG(
    "churn_pred_dag",
    default_args=default_args,
    schedule="0 6 1 * *",  # Run monthly on the 1st at 6AM
    catchup=False
) as dag:
    
    etl_operator = PythonOperator(
        task_id='etl_task',
        python_callable=etl_task
    )

    modelling_operator = PythonOperator(
        task_id='modelling_task',
        python_callable=modelling_task,
        provide_context=True
    )

    # mlflow_task = PythonOperator(
    #     task_id="mlflow_task",
    #     python_callable=mflow_log
    # )

    etl_operator >> modelling_operator
