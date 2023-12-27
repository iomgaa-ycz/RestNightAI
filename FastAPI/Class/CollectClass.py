from typing import List
from pydantic import BaseModel

class CCInputData(BaseModel):
    Time: str
    ID: str


class CCResponseData(BaseModel):
    ID: str
    action: int
    pose: int
    OnBed: int