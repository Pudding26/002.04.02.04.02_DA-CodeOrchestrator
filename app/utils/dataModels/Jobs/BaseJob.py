from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Union, Dict, ClassVar
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
import json
import math

from app.utils.dataModels.Jobs.util.RetryInfo import RetryInfo

class BaseJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_model: ClassVar[Optional[type]] = None  # REpresents the underlying orm_api this gets set in subclasses


    job_uuid : str = Field(default_factory=uuid4)
    job_type : str

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

    def to_orm(self, orm_cls):
        return orm_cls(**self.to_sql_row())

    def to_sql_row(self) -> dict:
        def _clean_nans(obj):
            """
            Recursively replace all float('nan'), inf, -inf with None in a nested structure.
            This ensures the object can be safely serialized to valid JSON.
            """
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, dict):
                return {k: _clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean_nans(item) for item in obj]
            return obj


        base_fields = {}
        raw_dict = self.model_dump(mode="json")     # Pydantic v2
        clean_dict = _clean_nans(raw_dict)
        for field_name, value in clean_dict.items():
            if isinstance(value, Enum):
                base_fields[field_name] = value.value
            elif isinstance(value, (UUID, datetime)):
                base_fields[field_name] = str(value)
            else:
                base_fields[field_name] = value

        # 🔄 Replaces compact model_dump_json() with pretty JSON string
        payload_pretty = json.dumps(
            self.model_dump(mode="json", exclude_none=False),
            indent=2,
            ensure_ascii=False
        )

        return {
            **base_fields,
            "payload": clean_dict,
        }


    @classmethod
    def from_sql_row(cls, row: dict) -> "BaseJob":
        return cls.model_validate(row["payload"])


    def update_db(self):
        """
        Update the job in the database using the linked api_model.
        """
        if self.api_model is None:
            raise ValueError(f"{self.__class__.__name__} must define `api_model` to support update_db()")

        row = self.to_sql_row()
        row["updated"] = str(datetime.utcnow())
        self.api_model.update_row(row)