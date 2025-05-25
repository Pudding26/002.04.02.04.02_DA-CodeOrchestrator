from sqlalchemy import Column, String, Integer
from app.utils.SQL.models.OrmBase import OrmBase

class DS40(OrmBase):
    __tablename__ = "DS40"

    genus = Column(String, primary_key=True)
    family = Column(String)
