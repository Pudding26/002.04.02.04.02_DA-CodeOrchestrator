from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Union, Dict
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

from app.utils.dataModels.Jobs.util.RetryInfo import RetryInfo
from app.utils.dataModels.Jobs.JobEnums import JobStatus

class BaseJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_uuid : str = Field(default_factory=uuid4)
    job_type : str
    segmenter_status: JobStatus = JobStatus.TODO
    modeler_status: JobStatus = JobStatus.TODO
    transfer_status: JobStatus = JobStatus.TODO
    loader_status: JobStatus = JobStatus.TODO
    created  : datetime = Field(default_factory=datetime.utcnow)
    updated  : datetime = Field(default_factory=datetime.utcnow)
    retry    : RetryInfo = Field(default_factory=RetryInfo)

    def register_failure(self, error: str):
        self.retry.register_failure(error)
        self.updated = datetime.utcnow()

    def is_ready(self, task: str) -> bool:
        task_field = f"{task}_status"
        if not hasattr(self, task_field):
            raise ValueError(f"Unknown task: {task}")

    def to_sql_row(self) -> dict:
        return {
            "job_uuid": str(self.job_uuid),
            "segmenter_status": self.segmenter_status.value,
            "modeler_status": self.modeler_status.value,
            "transfer_status": self.transfer_status.value,
            "loader_status": self.loader_status.value,
            "payload": self.model_dump(mode="json"),
            "created": self.created,
            "updated": self.updated
        }
    
    @classmethod
    def from_sql_row(cls, row: dict) -> "BaseJob":
        return cls.model_validate(row["payload"])

