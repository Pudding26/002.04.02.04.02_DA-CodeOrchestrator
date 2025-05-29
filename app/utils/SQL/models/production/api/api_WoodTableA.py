from typing import Optional, List, ClassVar, Any


from app.utils.SQL.models.api_BaseModel import api_BaseModel

from app.utils.SQL.models.production.orm.WoodTableA import WoodTableA  # Assuming this ORM class exists

class WoodTableA_Out(api_BaseModel):

    orm_class: ClassVar = WoodTableA 

    family: Optional[str]
    genus: Optional[str]
    species: Optional[str]
    IFAW_code: Optional[str]
    engName: Optional[str]
    deName: Optional[str]
    frName: Optional[str]
    japName: Optional[str]
    origin: Optional[str]
    sourceNo: Optional[str]

