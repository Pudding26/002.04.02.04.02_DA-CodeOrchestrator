from enum import Enum
from sqlalchemy.dialects.postgresql import UUID

from sqlalchemy import (
    Column,
    String,
    Enum as PgEnum,
    ForeignKey,
    Table,
    UUID as SQLUUID,
    create_engine,
)
from sqlalchemy.orm import relationship

from app.utils.dataModels.Jobs.JobEnums import JobKind, RelationState

from app.utils.SQL.models.orm_BaseModel import orm_BaseModel


class JobLink(orm_BaseModel):
    """
    Association-object pattern.
    We don’t declare a FK on child_uuid because it could live in any job table.
    """
    __tablename__  = "jobLink"

    parent_uuid = Column(String, ForeignKey("DoEJobs.job_uuid"), primary_key=True)
    child_uuid  = Column(String, primary_key=True)
    child_kind  = Column(PgEnum(JobKind), primary_key=True)
    rel_state   = Column(PgEnum(RelationState), default=RelationState.IN_PROGRESS)

    child_provider = relationship(
        "ProviderJobs",
        back_populates="parent_links",
        primaryjoin="foreign(JobLink.child_uuid) == ProviderJobs.job_uuid"
    )
    
    
    
    #parent_doe = relationship("DoEJobs", back_populates="doe_child_links")


    

    
