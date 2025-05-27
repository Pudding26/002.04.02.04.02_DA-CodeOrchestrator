from sqlalchemy import Column, String, Integer
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class DS09(orm_BaseModel):
    __tablename__ = "DS09"

    id = Column(Integer, primary_key=True, autoincrement=True)
    species = Column(String)
    IFAW_code = Column(String)
    origin = Column(String)
    engName = Column(String)
    deName = Column(String)
    frName = Column(String)
    genus = Column(String)
