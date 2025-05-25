from typing import Optional
from models.sharedBaseModel import SharedBaseModel

class SegmentationResultBase(SharedBaseModel):
    sampleID: str
    shotID: Optional[str]
    position: Optional[int]
    bin_type: Optional[str]
    percentile_bin: Optional[float]
    feature_name: Optional[str]
    stat_type: Optional[str]
    feature_value: Optional[float]
    unit: Optional[str]
    object_count: Optional[int]

class SegmentationResultOut(SegmentationResultBase):
    ROWID: int

    class Config:
        orm_mode = True
