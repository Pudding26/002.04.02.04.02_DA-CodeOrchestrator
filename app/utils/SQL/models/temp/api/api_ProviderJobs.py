from typing import List, ClassVar, Any
from app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.SQL.models.temp.orm.ProviderJobs import ProviderJobs

from datetime import datetime

class ProviderJobs_Out(api_BaseModel):
    orm_class: ClassVar = ProviderJobs
    db_key: ClassVar[str] = "temp"


    job_uuid: str
    og_job_uuids: list
    status: str
    payload: Any # is a JSNOB object
    created: datetime
    updated: datetime


