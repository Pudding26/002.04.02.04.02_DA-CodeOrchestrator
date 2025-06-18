from app.utils.SQL.models.orm_BaseModel import orm_BaseModel


from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy import DateTime

class ProviderJobs(orm_BaseModel):
    __tablename__ = "ProviderJobs"

    job_uuid = Column(String, primary_key=True)
    og_job_uuids = Column(JSONB)
    status = Column(String)
  
    payload = Column(JSONB)
    created = Column(DateTime)
    updated = Column(DateTime)
    