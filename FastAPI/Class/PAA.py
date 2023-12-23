from typing import List
from pydantic import BaseModel

class PAAInputData(BaseModel):
    PressureMap: List[str]
    ID: str


class PAAResponseData(BaseModel):
    ID: str
    action: int
    pose: int
    OnBed: int