from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "Dementia Progression Prediction API"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Model paths
    MMSE_MODEL_PATH: str = "app/models/MMSE_Model.keras"
    CDR_MODEL_PATH: str = "app/models/CDR_Model.keras"

    # Prediction settings
    DEFAULT_SEQUENCE_LENGTH: int = 2

    class Config:
        case_sensitive = True


settings = Settings()