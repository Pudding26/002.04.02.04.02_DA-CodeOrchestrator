from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Union, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum


from app.utils.dataModels.Jobs.BaseJob import BaseJob

class ProviderJobInput(BaseJob):
    job_type: str = "provider"
    job_no: int
    input: ProviderJobInput
    attrs: ProviderAttrs

class ProviderJobInput(BaseModel):
    src_file_path: str
    src_ds_rel_path: Union[str, List[str]]
    dest_rel_path: str
    stored_locally: List[int]
    image_data: Optional[Any] = None  # You could replace this with something serialisable or exclude

    model_config = ConfigDict(arbitrary_types_allowed=True)

class ProviderAttrs(BaseModel):
    Level1: Dict[str, Union[str, int]]
    Level2: Dict[str, Union[str, int]]
    Level3: Dict[str, Union[str, int]]
    Level4: Dict[str, Optional[str]]
    Level5: Dict[str, Union[str, int]]
    Level6: Dict[str, Optional[str]]
    Level7: Dict[str, Union[str, int, float, Optional[str]]]
    dataSet_attrs: Dict[str, Optional[Union[str, float, int]]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

