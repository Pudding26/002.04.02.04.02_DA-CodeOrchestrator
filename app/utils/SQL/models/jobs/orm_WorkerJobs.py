from app.utils.SQL.models.orm_BaseModel import orm_BaseModel
from app.utils.SQL.DBEngine import DBEngine


from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, Integer
from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy import DateTime


from sqlalchemy import update
from contextlib import contextmanager

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
        back_populates="child",
        primaryjoin="orm_WorkerJobs.job_uuid == foreign(JobLink.child_uuid)"
    )


    @classmethod
    def update_row(cls, row: dict):
        # Use a context manager from your DB layer
        session = DBEngine("jobs").get_session()
        job_uuid = row.get("job_uuid")

        if not job_uuid:
            raise ValueError("Missing job_uuid for update.")

        stmt = (
            update(cls)
            .where(cls.job_uuid == job_uuid)
            .values(**row)
        )

        session.execute(stmt)
        session.commit()


