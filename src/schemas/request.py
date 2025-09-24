from pydantic import BaseModel
from .predictions import Features
from typing import List

class BatchRequest(BaseModel):
    features: List[Features]