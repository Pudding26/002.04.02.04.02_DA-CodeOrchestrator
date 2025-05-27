from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class WoodMaster(orm_BaseModel):
    __tablename__ = "woodMaster"

    sampleID = Column(String, primary_key=True)

    family = Column(String)
    genus = Column(String)
    species = Column(String)
    totalNumberShots = Column(Integer)
    lens = Column(String)
    view = Column(String)
    woodType = Column(String)
    institution = Column(String)
    contributor = Column(String)
    digitizedDate = Column(DateTime)
    microscopicTechnic = Column(String)
    resolution = Column(Float)
    DPI = Column(Integer)
    bitDepth = Column(Integer)
    colorDepth = Column(Integer)
    colorSpace = Column(String)
    sourceID = Column(String)
    sourceNo = Column(String)
    specimenID = Column(String)
    deName = Column(String)
    engName = Column(String)
    frName = Column(String)
    japName = Column(String)
    path = Column(String)
    filterNo = Column(Integer)
    stackID = Column(String)

    # ✅ Add relationship to segmentationResults
    segmentation_results = relationship(
        "SegmentationResults",
        back_populates="wood_sample",
        cascade="all, delete-orphan"
    )
