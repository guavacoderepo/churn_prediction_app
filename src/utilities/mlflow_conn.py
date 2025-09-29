import dagshub
import dagshub.auth
import mlflow
from config.setting import Settings

def mlflow_connect():
    """Connect to MLflow, create experiment if missing, and return experiment ID."""
    try:
        settings = Settings()  # type: ignore
        tracking_uri = settings.TRACKING_URI
        experiment_name = settings.EXPERIMENT_NAME
        repo_name = settings.REPO_NAME
        repo_ower = settings.REPO_OWNER
        token = settings.DAGSHUB_TOKEN
        
        dagshub.auth.add_app_token(token)  # type: ignore
        
        dagshub.init(repo_owner=repo_ower, repo_name=repo_name, mlflow=True) # type: ignore
        mlflow.set_tracking_uri(f"https://{tracking_uri}")
        exp = mlflow.get_experiment_by_name(experiment_name) # type: ignore

        if exp is None:
            mlflow.create_experiment(
                name= experiment_name,
                tags={
                    "mlflow.note.content": "Customer churn prediction pipeline",
                    "team": "AI engineering team",
                    "project": "churn_prediction_end_to_end"
                }
            )
            exp = mlflow.get_experiment_by_name(experiment_name)
            print("✅ Created new experiment:", exp.experiment_id) # type: ignore
        if exp is None:
            raise RuntimeError("❌ Failed to create or fetch MLflow experiment.")
        return exp.experiment_id # type: ignore

    except Exception as e:
         raise ConnectionError(f"Failed to initialize DagsHub or connect to MLflow: {e}") from e
