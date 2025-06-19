from app.utils.SQL.models.orm_BaseModel import orm_BaseModel
from sqlalchemy.orm import relationship

from sqlalchemy import Column, String, Integer
from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy import DateTime

class orm_WorkerJobs(orm_BaseModel):
    __tablename__ = "WorkerJobs"

    job_uuid = Column(String, primary_key=True)
    job_type = Column(String)
    
    status = Column(String)
    attempts = Column(Integer)
    next_retry = Column(DateTime)
    
    created = Column(DateTime)
    updated = Column(DateTime)
    
    payload = Column(JSONB)
    parent_job_uuids = Column(JSONB)



    parent_links = relationship(
    "JobLink",
    back_populates="child_provider",
    primaryjoin="ProviderJobs.job_uuid == foreign(JobLink.child_uuid)"
    )
