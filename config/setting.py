from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",             # load from .env file
        env_file_encoding="utf-8",   # encoding
        extra="ignore"               # ignore unknown env vars
    )

    TRACKING_URI: str
    RUN_ID:str
    EXPERIMENT_NAME:str
    SECRET_KEY: str
    DB_PATH: str
    MODEL_URI:str
    env: str = "development"