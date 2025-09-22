from pydantic import BaseModel
from typing import Optional

class Features(BaseModel):
    customerID: Optional[str]
    gender: str
    SeniorCitizen:int
    Partner: str
    Dependents:str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionResult(BaseModel):
    customerID: Optional[str]
    churn_score: float
    churn: str
