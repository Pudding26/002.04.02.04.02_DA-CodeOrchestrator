from sqlalchemy import Column, String, Integer
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class DS40(orm_BaseModel):
    __tablename__ = "DS40"

    genus = Column(String, primary_key=True)
    family = Column(String)
