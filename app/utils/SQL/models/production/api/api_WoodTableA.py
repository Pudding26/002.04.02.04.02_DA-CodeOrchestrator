from typing import Optional


from app.utils.SQL.models.api_BaseModel import api_BaseModel


class WoodTableABase(api_BaseModel):
    family: Optional[str]
    genus: Optional[str]
    species: Optional[str]
    IFAW_ID: Optional[str]
    engName: Optional[str]
    deName: Optional[str]
    frName: Optional[str]
    japName: Optional[str]
    origin: Optional[str]
    sourceNo: Optional[str]

