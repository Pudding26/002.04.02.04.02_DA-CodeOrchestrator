from typing import List, ClassVar, Any
from app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.SQL.models.temp.orm.DoEJobs import DoEJobs

from datetime import datetime

class DoEJobs_Out(api_BaseModel):
    orm_class: ClassVar = DoEJobs
    db_key: ClassVar[str] = "temp"


    job_uuid: str
    segmenter_status: str
    modeler_status: str
    transfer_status: str
    provider_status: str
    payload: dict
    created: datetime
    updated: datetime


