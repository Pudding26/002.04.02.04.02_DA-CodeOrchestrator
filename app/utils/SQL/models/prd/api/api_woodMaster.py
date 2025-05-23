from typing import Optional, List
from datetime import datetime
from models.sharedBaseModel import SharedBaseModel
from models.prd.api.api_segmentationResults import SegmentationResultOut

class WoodMasterBase(SharedBaseModel):
    family: Optional[str]
    genus: Optional[str]
    species: Optional[str]
    totalNumberShots: Optional[int]
    lens: Optional[str]
    view: Optional[str]
    woodType: Optional[str]
    institution: Optional[str]
    contributor: Optional[str]
    digitizedDate: Optional[datetime]
    microscopicTechnic: Optional[str]
    resolution: Optional[float]
    DPI: Optional[int]
    bitDepth: Optional[int]
    colorDepth: Optional[int]
    colorSpace: Optional[str]
    sourceID: Optional[str]
    sourceNo: Optional[str]
    specimenID: Optional[str]
    deName: Optional[str]
    engName: Optional[str]
    frName: Optional[str]
    japName: Optional[str]
    path: Optional[str]
    filterNo: Optional[int]
    stackID: Optional[str]

class WoodMasterOut(WoodMasterBase):
    sampleID: str
    segmentation_results: Optional[List[SegmentationResultOut]] = []

    class Config:
        orm_mode = True
