from pydantic import BaseModel
from ..schemas.predictions import PredictionResult
from typing import List

class BatchResponse(BaseModel):
    result: List[PredictionResult]
