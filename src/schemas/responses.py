from pydantic import BaseModel
from .predictions import PredictionResult
from typing import List

class BatchResponse(BaseModel):
    result: List[PredictionResult]
