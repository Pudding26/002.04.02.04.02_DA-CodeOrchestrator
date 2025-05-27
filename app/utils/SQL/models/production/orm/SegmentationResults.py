from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class SegmentationResultsAPI(orm_BaseModel):
    __tablename__ = "segmentationResults"

    ROWID = Column(Integer, primary_key=True, autoincrement=True)

    sampleID = Column(String, ForeignKey("woodMaster.sampleID"), nullable=False)
    shotID = Column(String)
    position = Column(Integer)

    bin_type = Column(String)
    percentile_bin = Column(Float)

    feature_name = Column(String)
    stat_type = Column(String)
    feature_value = Column(Float)

    unit = Column(String)
    object_count = Column(Integer)

    # ✅ Add reverse relationship
    wood_sample = relationship("WoodMaster", back_populates="segmentation_results")
