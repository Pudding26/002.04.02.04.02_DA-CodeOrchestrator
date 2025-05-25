from sqlalchemy import Column, String, Integer
from app.utils.SQL.models.OrmBase import OrmBase

class WoodTableA(OrmBase):
    __tablename__ = "WoodTableA"

    id = Column(Integer, primary_key=True, autoincrement=True)
    family = Column(String)
    genus = Column(String)
    species = Column(String)
    IFAW_ID = Column(String)
    engName = Column(String)
    deName = Column(String)
    frName = Column(String)
    japName = Column(String)
    origin = Column(String)
    sourceNo = Column(String)
