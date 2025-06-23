from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

#ORM
from app.utils.SQL.models.jobs.orm_JobLink import orm_JobLink



from app.utils.dataModels.Jobs.BaseJob import BaseJob
from app.utils.dataModels.Jobs.DoEJob import DoEJob

from app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind, RelationState

from app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs

from uuid import UUID

from sqlalchemy import event, delete, insert, update, select, func
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapper




class SegmenterJob(BaseJob):
    
    
    job_type: JobKind = JobKind.PROVIDER
    orm_model = orm_WorkerJobs
    status: JobStatus = JobStatus.TODO.value
    input: SegmenterJobInput
    attrs: Optional[Dict[str, Any]] = None
    

    model_config = ConfigDict(extra="forbid")


class SegmenterJobInput(BaseModel):
    hdf5_path: str = "data/productionData/primaryData.hdf5"
    src_file_path: str
    
    dest_GS_file_path: str
    dest_FF_file_path: str
    
    dest_FF_stackID: str
    dest_GS_stackID: str

    dest_FilterNo: str
    


    job_No: Optional[int] = None


    model_config = ConfigDict(arbitrary_types_allowed=True)


