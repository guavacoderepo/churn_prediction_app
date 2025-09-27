from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",             # load from .env file
        env_file_encoding="utf-8",   # encoding
        extra="ignore"               # ignore unknown env vars
    )

    TRACKING_URI: str
    RUN_NAME:str
    EXPERIMENT_NAME:str
    AIRFLOW_UID: str
    MODEL_URI:str
    MODEL_NAME:str
    DAGSHUB_TOKEN:str

    env: str = "development"