from sqlalchemy import Column, String, Integer
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class DS12(orm_BaseModel):
    __tablename__ = "DS12"

    species = Column(String, primary_key=True)
    engName = Column(String)
    deName = Column(String)
