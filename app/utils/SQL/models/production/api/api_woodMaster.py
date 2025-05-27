from typing import Optional, List
from datetime import datetime
from app.utils.SQL.models.api_BaseModel import api_BaseModel


class WoodMasterBase(api_BaseModel):
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

