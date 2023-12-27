from typing import List
from pydantic import BaseModel
import datetime
from typing import Optional

class CCInputData(BaseModel):
    Time: str
    ID: str


class CCResponseData(BaseModel):
    ID: str
    action: int
    pose: int
    OnBed: int

class CCRecordData(BaseModel):
    ID: str
    begin_time: datetime.datetime
    end_time: Optional[datetime.datetime]