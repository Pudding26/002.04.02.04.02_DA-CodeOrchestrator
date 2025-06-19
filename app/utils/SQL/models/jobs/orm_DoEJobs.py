from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

from sqlalchemy.orm import relationship
from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy import DateTime

class orm_DoEJobs(orm_BaseModel):
    __tablename__ = "DoEJobs"

    job_uuid = Column(String, primary_key=True)
    segmenter_status = Column(String)
    modeler_status = Column(String)
    transfer_status = Column(String)
    provider_status = Column(String)
    payload = Column(JSONB)
    created = Column(DateTime)
    updated = Column(DateTime)
    
    #doe_child_links = relationship("JobLink", back_populates="parent_doe", foreign_keys="JobLink.parent_uuid")
